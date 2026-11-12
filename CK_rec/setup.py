import re
import rtmidi


def device(port):
    """A port's name without its ALSA client number, which can change when the device is plugged in again."""
    return re.sub(r' \d+:\d+$', '', port)

class Setup(object):

    def __init__(self):
        self.__midiin = rtmidi.MidiIn()
        self.__ports = self.__midiin.get_ports()

    def show_ports(self):
        print("These are your detected MIDI devices:", '\n')
        for port in self.__ports:
            print(self.__ports.index(port), " -> ", port)

    def get_port_from_user(self):
        selected_midiport = -1
        while selected_midiport < 0:
            try:
                choice = input("Please choose the MIDI device (number) you want to use and hit Enter "
                               "(just Enter: none, you play the laptop's keyboard in the piano roll):")
                if not choice.strip():
                    return None
                selected_midiport = int(choice)
                if selected_midiport < 0 or selected_midiport >= len(self.__ports):
                    print("Invalid number, please try again:")
                    selected_midiport = -1
                else:
                    return selected_midiport
            except KeyboardInterrupt:
                print('\n', "You want to quit? ¯\('…')/¯  ok, Bye bye.")
                exit()
            except ValueError:
                print("Sorry, type a valid port numer!")

    def open_port(self, pnum):
        print("You have chosen: ", self.__ports[pnum])

        if self.__ports:
            #TODO: do we need to check on the existence of ports?
            self.__midiin.open_port(pnum)
            # ignore sysex, timing and active sense messages
            self.__midiin.ignore_types(True, True, False)
        else:
            raise Exception("No midi ports! Maybe open a virtual device?")

    def close_port(self):
        self.__midiin.close_port()

    def set_callback(self,cb):
        self.__callback = cb
        self.__midiin.set_callback(cb)

    def port_name(self, pnum):
        return self.__ports[pnum]

    def present(self, name):
        """Whether the device of the port `name` is plugged in."""
        return device(name) in map(device, self.__midiin.get_ports())

    def reopen(self, name):
        """Opens the device of the port `name` again, after it was unplugged and plugged back in."""
        ports = self.__midiin.get_ports()
        pnum = next(i for i, p in enumerate(ports) if device(p) == device(name))
        self.__midiin.close_port()
        self.__midiin.open_port(pnum)
        self.__midiin.ignore_types(True, True, False)
        self.__midiin.set_callback(self.__callback)
        print("Plugged in again:", ports[pnum])

    def perform_setup(self):
        self.show_ports()
        myPort = self.get_port_from_user()
        return myPort

    def end(self):
        print("Bye bye 🎹\n")
        self.close_port()
        del self.__midiin
