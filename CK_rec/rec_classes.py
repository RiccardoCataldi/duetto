import rtmidi

class CK_rec(object):
    def __init__(self):
        # Initialize channel assignments first
        self.input_channel = 0  # Channel for keyboard input
        self.generated_channel = 1  # Channel for generated music
        
        # Then set up MIDI output
        self.setup_output()
    
    def setup_output(self):
        self.__midiout = rtmidi.MidiOut()
        self.__ports_out = self.__midiout.get_ports()
        if self.__ports_out:
            self.__midiout.open_port(0)  # Open the first available MIDI output device
            print(f"MIDI output opened on: {self.__ports_out[0]}")
            # Send initial program changes
            self.__midiout.send_message([0xC0 + self.input_channel, 0])  # Piano
            self.__midiout.send_message([0xC0 + self.generated_channel, 48])  # Strings
    
    def play_midi_message(self, message, is_generated=False, channel=None):
        if self.__midiout and self.__midiout.is_port_open():
            try:
                # Modify the channel based on whether it's generated or input, unless given
                if channel is None:
                    channel = self.generated_channel if is_generated else self.input_channel
                # Convert to channel message
                if message[0] & 0xF0 in [0x80, 0x90]:  # Note on/off
                    msg = [message[0] & 0xF0 | channel, message[1], message[2]]
                else:
                    msg = message
                self.__midiout.send_message(msg)
            except Exception as e:
                print(f"Error sending MIDI message: {e}")
    
    def set_input_instrument(self, program):
        """Change the input instrument"""
        if 0 <= program <= 127:
            self.input_channel = 0
            self.__midiout.send_message([0xC0 + self.input_channel, program])
            print(f"Input instrument changed to program {program}")
    
    def set_generated_instrument(self, program):
        """Change the generated music instrument"""
        if 0 <= program <= 127:
            self.generated_channel = 1
            self.__midiout.send_message([0xC0 + self.generated_channel, program])
            print(f"Generated instrument changed to program {program}")
