"""Canonical MIDI note containers."""

from dataclasses import dataclass
from pathlib import Path
import warnings

from mido import Message, MetaMessage, MidiFile, MidiTrack, bpm2tempo, second2tick

SAVE_BPM = 120
SAVE_TICKS_PER_BEAT = 480


@dataclass(slots=True)
class Note:
    """A note with start and duration measured in seconds."""

    is_drum: bool
    program: int
    pitch: int
    velocity: int
    start: float
    duration: float


@dataclass(slots=True)
class Midi:
    """Canonical MIDI data containing notes only."""

    notes: list[Note]

    @classmethod
    def load(cls, midi_path: str | Path) -> "Midi":
        """Load notes using Mido's tempo-aware playback timeline."""
        path = Path(midi_path)
        source = MidiFile(filename=str(path))
        notes: list[Note] = []
        programs = [0] * 16
        active: dict[tuple[int, int], tuple[float, int, int]] = {}
        time = 0.0
        zero_length_notes = 0
        unmatched_note_offs = 0

        for event in source:
            time += event.time
            if not isinstance(event, Message):
                continue
            if event.type == "program_change":
                programs[event.channel] = event.program
                continue
            if event.type not in {"note_on", "note_off"}:
                continue

            key = (event.channel, event.note)
            is_note_on = event.type == "note_on" and event.velocity > 0

            if is_note_on:
                previous = active.get(key)
                if previous is not None:
                    start, velocity, program = previous
                    if time == start:
                        zero_length_notes += 1
                    else:
                        notes.append(
                            Note(
                                pitch=event.note,
                                velocity=velocity,
                                is_drum=event.channel == 9,
                                program=program,
                                start=start,
                                duration=time - start,
                            )
                        )
                active[key] = (
                    time,
                    event.velocity,
                    0 if event.channel == 9 else programs[event.channel],
                )
                continue

            previous = active.pop(key, None)
            if previous is None:
                unmatched_note_offs += 1
                continue

            start, velocity, program = previous
            if time == start:
                zero_length_notes += 1
                continue
            notes.append(
                Note(
                    pitch=event.note,
                    velocity=velocity,
                    is_drum=event.channel == 9,
                    program=program,
                    start=start,
                    duration=time - start,
                )
            )

        unmatched_note_ons = len(active)
        notes.sort(
            key=lambda note: (
                note.start,
                note.duration,
                note.is_drum,
                note.program,
                note.pitch,
                note.velocity,
            )
        )

        if zero_length_notes or unmatched_note_offs or unmatched_note_ons:
            warnings.warn(
                f"{path}: dropped MIDI note events "
                f"(zero-length notes={zero_length_notes}, "
                f"unmatched note-offs={unmatched_note_offs}, "
                f"unmatched note-ons={unmatched_note_ons})",
                UserWarning,
                stacklevel=2,
            )

        return cls(notes=notes)

    def save(self, midi_path: str | Path) -> None:
        """Save notes as a fresh 120 BPM, 480-ticks-per-beat MIDI file."""
        melodic_programs = sorted(
            {note.program for note in self.notes if not note.is_drum}
        )
        melodic_channels = [*range(9), *range(10, 16)]
        if len(melodic_programs) > len(melodic_channels):
            raise ValueError(
                "Cannot save more than 15 distinct melodic programs without "
                "recorded channel assignments"
            )
        channel_by_program = dict(zip(melodic_programs, melodic_channels))
        tempo = bpm2tempo(SAVE_BPM)
        scheduled: list[tuple[int, int, int, Message | MetaMessage]] = [
            (0, 0, 0, MetaMessage("set_tempo", tempo=tempo, time=0))
        ]

        order_offset = 1
        for order, (program, channel) in enumerate(channel_by_program.items()):
            if program == 0:
                continue
            scheduled.append(
                (
                    0,
                    1,
                    order_offset + order,
                    Message(
                        "program_change",
                        channel=channel,
                        program=program,
                        time=0,
                    ),
                )
            )

        order_offset += len(melodic_programs)
        for order, note in enumerate(self.notes, start=order_offset):
            if note.start < 0:
                raise ValueError("Note start must be non-negative")
            if note.duration <= 0:
                raise ValueError("Note duration must be positive")

            channel = 9 if note.is_drum else channel_by_program[note.program]
            start_tick = round(
                second2tick(note.start, SAVE_TICKS_PER_BEAT, tempo)
            )
            end_tick = round(
                second2tick(
                    note.start + note.duration,
                    SAVE_TICKS_PER_BEAT,
                    tempo,
                )
            )
            end_tick = max(end_tick, start_tick + 1)
            scheduled.extend(
                [
                    (
                        start_tick,
                        2,
                        order,
                        Message(
                            "note_on",
                            channel=channel,
                            note=note.pitch,
                            velocity=note.velocity,
                            time=0,
                        ),
                    ),
                    (
                        end_tick,
                        0,
                        order,
                        Message(
                            "note_off",
                            channel=channel,
                            note=note.pitch,
                            velocity=0,
                            time=0,
                        ),
                    ),
                ]
            )

        restored = MidiFile(type=0, ticks_per_beat=SAVE_TICKS_PER_BEAT)
        track = MidiTrack()
        previous_tick = 0
        for tick, _, _, message in sorted(scheduled, key=lambda item: item[:3]):
            track.append(message.copy(time=tick - previous_tick))
            previous_tick = tick
        restored.tracks.append(track)
        restored.save(filename=str(midi_path))
