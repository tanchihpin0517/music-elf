"""Tests for canonical MIDI note parsing."""

from dataclasses import fields
from pathlib import Path
import warnings

from mido import Message, MetaMessage, MidiFile, MidiTrack, bpm2tempo
import pytest

from music_elf import GigaMidiDataset, Midi, Note


def save_midi(path: Path, *tracks: MidiTrack, ticks_per_beat: int = 480) -> None:
    midi = MidiFile(ticks_per_beat=ticks_per_beat)
    midi.tracks.extend(tracks)
    midi.save(path)


def assert_note_time(note: Note, start: float, duration: float) -> None:
    assert note.start == pytest.approx(start)
    assert note.duration == pytest.approx(duration)


def test_midi_contains_only_notes() -> None:
    assert [field.name for field in fields(Midi)] == ["notes"]


def test_midi_load_normalizes_both_note_off_forms(tmp_path: Path) -> None:
    track = MidiTrack(
        [
            Message("note_on", channel=2, note=60, velocity=100, time=0),
            Message("note_off", channel=2, note=60, velocity=45, time=240),
            Message("note_on", channel=2, note=61, velocity=80, time=120),
            Message("note_on", channel=2, note=61, velocity=0, time=240),
        ]
    )
    path = tmp_path / "notes.mid"
    save_midi(path, track)

    midi = Midi.load(path)

    assert midi.notes[:1] == [
        Note(False, 0, 60, 100, start=0.0, duration=0.25)
    ]
    assert midi.notes[1].pitch == 61
    assert midi.notes[1].velocity == 80
    assert_note_time(midi.notes[1], start=0.375, duration=0.25)


def test_midi_load_converts_tempo_changes_to_seconds(tmp_path: Path) -> None:
    track = MidiTrack(
        [
            MetaMessage("set_tempo", tempo=bpm2tempo(120), time=0),
            Message("note_on", note=60, velocity=100, time=0),
            Message("note_off", note=60, velocity=0, time=480),
            MetaMessage("set_tempo", tempo=bpm2tempo(60), time=0),
            Message("note_on", note=61, velocity=100, time=0),
            Message("note_off", note=61, velocity=0, time=480),
        ]
    )
    path = tmp_path / "tempo.mid"
    save_midi(path, track)

    midi = Midi.load(path)

    assert_note_time(midi.notes[0], start=0.0, duration=0.5)
    assert_note_time(midi.notes[1], start=0.5, duration=1.0)


def test_midi_load_preserves_programs_and_sorts_notes(tmp_path: Path) -> None:
    later = MidiTrack(
        [
            Message("program_change", channel=0, program=73, time=0),
            Message("note_on", channel=0, note=72, velocity=90, time=480),
            Message("note_off", channel=0, note=72, velocity=0, time=240),
        ]
    )
    earlier = MidiTrack(
        [
            Message("program_change", channel=1, program=46, time=0),
            Message("note_on", channel=1, note=48, velocity=70, time=240),
            Message("note_off", channel=1, note=48, velocity=0, time=240),
        ]
    )
    path = tmp_path / "programs.mid"
    save_midi(path, later, earlier)

    midi = Midi.load(path)

    assert [(note.pitch, note.program) for note in midi.notes] == [
        (48, 46),
        (72, 73),
    ]
    assert_note_time(midi.notes[0], start=0.25, duration=0.25)
    assert_note_time(midi.notes[1], start=0.5, duration=0.25)


def test_repeated_note_on_closes_active_note(tmp_path: Path) -> None:
    track = MidiTrack(
        [
            Message("note_on", note=60, velocity=100, time=0),
            Message("note_on", note=60, velocity=80, time=240),
            Message("note_off", note=60, velocity=0, time=240),
        ]
    )
    path = tmp_path / "repeat.mid"
    save_midi(path, track)

    midi = Midi.load(path)

    assert [note.velocity for note in midi.notes] == [100, 80]
    assert_note_time(midi.notes[0], start=0.0, duration=0.25)
    assert_note_time(midi.notes[1], start=0.25, duration=0.25)


def test_parser_warns_once_for_dropped_events(tmp_path: Path) -> None:
    track = MidiTrack(
        [
            Message("note_off", note=40, velocity=0, time=0),
            Message("note_on", note=41, velocity=90, time=0),
            Message("note_off", note=41, velocity=0, time=0),
            Message("note_on", note=42, velocity=90, time=10),
        ]
    )
    path = tmp_path / "malformed.mid"
    save_midi(path, track)

    with pytest.warns(
        UserWarning,
        match=(
            r"zero-length notes=1, unmatched note-offs=1, "
            r"unmatched note-ons=1"
        ),
    ) as caught:
        midi = Midi.load(path)

    assert len(caught) == 1
    assert midi.notes == []


def test_clean_file_emits_no_warning(tmp_path: Path) -> None:
    track = MidiTrack(
        [
            Message("note_on", note=60, velocity=100, time=0),
            Message("note_off", note=60, velocity=0, time=480),
        ]
    )
    path = tmp_path / "clean.mid"
    save_midi(path, track)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        midi = Midi.load(path)

    assert len(midi.notes) == 1


def test_midi_load_is_a_dataset_transform(tmp_path: Path) -> None:
    track = MidiTrack(
        [
            Message("note_on", note=60, velocity=100, time=0),
            Message("note_off", note=60, velocity=0, time=480),
        ]
    )
    path = tmp_path / "sample.mid"
    save_midi(path, track)
    (tmp_path / "files.txt").write_text("sample.mid\n", encoding="utf-8")

    midi = GigaMidiDataset(tmp_path, transform=Midi.load)[0]

    assert isinstance(midi, Midi)
    assert_note_time(midi.notes[0], start=0.0, duration=0.5)


def test_midi_load_marks_drums(tmp_path: Path) -> None:
    track = MidiTrack(
        [
            Message("note_on", channel=9, note=36, velocity=100, time=0),
            Message("note_off", channel=9, note=36, velocity=0, time=240),
            Message("note_on", channel=8, note=60, velocity=100, time=0),
            Message("note_off", channel=8, note=60, velocity=0, time=240),
        ]
    )
    path = tmp_path / "drums.mid"
    save_midi(path, track)

    midi = Midi.load(path)

    assert [note.is_drum for note in midi.notes] == [True, False]
    assert [note.program for note in midi.notes] == [0, 0]


def test_midi_save_restores_notes_and_programs(tmp_path: Path) -> None:
    midi = Midi(
        notes=[
            Note(
                pitch=60,
                velocity=100,
                is_drum=False,
                program=41,
                start=0.125,
                duration=0.375,
            ),
            Note(
                pitch=36,
                velocity=110,
                is_drum=True,
                program=0,
                start=0.25,
                duration=0.125,
            ),
        ]
    )
    path = tmp_path / "restored.mid"

    midi.save(path)
    raw = MidiFile(path)
    restored = Midi.load(path)

    assert raw.type == 0
    assert raw.ticks_per_beat == 480
    assert any(event.type == "set_tempo" for event in raw.tracks[0])
    assert any(
        event.type == "program_change" and event.program == 41
        for event in raw.tracks[0]
    )
    assert len(restored.notes) == 2
    for actual, expected in zip(restored.notes, midi.notes):
        assert actual.pitch == expected.pitch
        assert actual.velocity == expected.velocity
        assert actual.is_drum == expected.is_drum
        assert actual.program == expected.program
        assert_note_time(actual, expected.start, expected.duration)
