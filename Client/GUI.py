from tkinter import *
from tkinter import messagebox
from tkinter.ttk import *
import threading
from functools import partial
import ipaddress

from time import sleep


class ConnectSelector:
    def __init__(self, rpidict, last):
        self.geometry = "400x200"
        self.title = "Select"
        self.root = Tk()
        self.root.protocol("WM_DELETE_WINDOW", self.callback_exit)
        self.quitFlag = False
        self.root.title(self.title)
        self.root.geometry(self.geometry)

        self.rpidict = rpidict
        # self.rpilist = list(rpidict)
        self.combo = Combobox(values=list(self.rpidict) + ["<new>"], state="readonly")
        self.combo.current(list(self.rpidict).index(last))
        self.combo.bind("<<ComboboxSelected>>", self.callback_combo)
        self.combo.pack()

        self.delete = Button(text="delete", command=self.callback_delete)
        self.delete.pack()

        self.OK = Button(text="OK", command=self.callback_OK)
        self.OK.pack()

        self.IPEntrys = []
        self.LabIPin = Label(text="IP:").pack(side=LEFT, padx=2)

        for e in range(0, 4):
            self.IPEntrys.append(Entry(width=3))
            self.IPEntrys[e].pack(side=LEFT)
            if e < 3:
                self.tecka = Label(text=".").pack(side=LEFT)

        self.dvojtecka = Label(text=":").pack(side=LEFT)
        self.PortIn = Entry(width=5)
        self.PortIn.pack(side=LEFT)

        self.namelab = Label(text="  název:").pack(side=LEFT)
        self.name = Entry()
        self.name.pack(side=LEFT)

        self.SAVE = Button(text="SAVE", command=self.callback_SAVE)
        self.SAVE.pack(side=BOTTOM)

        self.callback_combo()

        self.root.mainloop()

    def callback_combo(self, _=None):
        selected = self.combo.get()

        for e in self.IPEntrys:
            e.delete(0, "end")
        self.PortIn.delete(0, "end")
        self.name.delete(0, "end")

        if selected != "<new>":
            ip, port = self.rpidict[selected].split(":")
            for i, e in enumerate(ip.split(".")):
                self.IPEntrys[i].insert(0, e)
            self.PortIn.insert(0, port)
            self.name.insert(0, selected)

    def callback_delete(self):
        selected = self.combo.get()
        if selected != "<new>":
            self.rpidict.pop(self.combo.get())
            self.combo.configure(values=list(self.rpidict) + ["<new>"])
            self.combo.current(0)

    def callback_OK(self):
        self.IP_PORT = self.ip_parse()
        self.laststate = self.combo.get()
        if self.IP_PORT is None:
            return
        self.root.destroy()
        self.root.quit()

    def callback_SAVE(self):
        selected = self.combo.get()
        name = self.name.get()
        ip_port = self.ip_parse()
        if ip_port is None:
            return
        else:
            ip, port = ip_port
            port = str(port)
        if len(name) == 0 or " " in name or name=="<new>":
            messagebox.showerror("Bad name", "Název není ve správném formátu!")
            return
        if selected != "<new>":
            self.rpidict.pop(selected)

        self.rpidict[name] = ":".join([ip, port])
        self.combo.configure(values=list(self.rpidict) + ["<new>"])
        self.combo.current(list(self.rpidict).index(name))

    def ip_parse(self):
        IP = ""
        for e in self.IPEntrys:
            IP += e.get() + "."
        IP = IP[:-1]

        try:
            ipaddress.ip_address(IP)
        except ValueError:
            messagebox.showerror("Bad IP", "IP adresa není ve správném formátu!")
            return None
        try:
            Port = int(self.PortIn.get())
        except ValueError:
            messagebox.showerror("Bad PORT", "PORT není ve správném formátu!")
            return None
        if not 0 <= Port <= 65535:
            messagebox.showerror("Bad PORT", "Špatné číslo portu!")
            return None

        return IP, Port
    #
    # def combostate(self):
    #     print(self.combo.get())
    #     return "localhost"

    def callback_exit(self):
        self.root.quit()
        self.quitFlag = True


class Vizualizace(threading.Thread):
    def __init__(self, geometry, title):
        threading.Thread.__init__(self)
        self.geometry = geometry
        self.title = title
        self.refreshGUI = False
        self.refreshIMG = False
        self.saveCommand = False

        self.ScaleValues = {}

        self.quitFlag = False
        self.start()

    def startGUI(self, images, curr_config_data, functions):

        self.pictures = images

        self.images = []
        self.scales = []
        self.combos = []
        self.labels = []

        def scalecallback(var, val):
            # print(val, var)
            self.ScaleValues[var] = int(float(val))

        def combocallback(which, _):
            var = self.combos[which].get()
            if var == "None" and len(self.combos) == which+2:
                self.order.pop(-1)
            elif var != "None" and len(self.combos) == which+1:
                self.order.append(var)
            elif var != "None" and len(self.combos) >= which+2:
                self.order[which] = var
            else:
                return
            for items in self.images + self.scales + self.combos + self.labels:
                items.destroy()
                self.TopFrame.destroy()
                self.MiddleFrame.destroy()
            self.refreshGUI = True

        def buttoncallback():
            self.saveCommand = True

        self.TopFrame = Frame(self.root, width=10, height=10)
        self.TopFrame.grid(row=0, column=0, sticky=W, pady=20)
        self.MiddleFrame = Frame(self.root, width=10, height=10)
        self.MiddleFrame.grid(row=1, column=0, sticky=W, pady=20)
        self.BottomFrame = Frame(self.root, width=10, height=10)
        self.BottomFrame.grid(row=2, column=0, sticky=W, pady=20)

        self.label1 = Label(self.TopFrame, text="original image")
        self.label1.grid(row=0, column=0)
        self.images.append(Label(self.TopFrame, image=self.pictures[0]))
        self.images[0].grid(row=1, column=0)

        self.order = curr_config_data["order"]

        listOfFunctions = list(functions)

        frame = self.TopFrame
        vrch = True
        nest_idx = 0
        col = 1
        row = 1
        for i, f in enumerate(self.order):
            for a in functions[f]:
                self.scales.append(Scale(frame, value=curr_config_data[a],
                                         orient=VERTICAL, from_=1020, to=0, length=300,
                                         command=partial(scalecallback, a)))
                self.scales[nest_idx].grid(row=row, column=col)

                self.labels.append(Label(frame, text=a))
                self.labels[nest_idx].grid(row=row+1, column=col)

                nest_idx += 1
                col += 1

            self.images.append(Label(frame, image=self.pictures[i+1]))
            self.images[i+1].grid(row=row, column=col)
            if i == len(self.order)-1:
                listOfFunctions = listOfFunctions+["None"]
            self.combos.append(Combobox(frame, values=listOfFunctions, state="readonly"))
            self.combos[i].bind("<<ComboboxSelected>>", partial(combocallback, i))
            self.combos[i].current(listOfFunctions.index(self.order[i]))
            self.combos[i].grid(row=row-1, column=col)

            col += 1

            if i > 2 and vrch:
                col = 0
                row = 1
                frame = self.MiddleFrame
                vrch = False

        if len(self.pictures) < 10:
            extra = len(self.order)
            self.combos.append(Combobox(frame, values=list(functions) + ["None"], state="readonly"))
            self.combos[extra].bind("<<ComboboxSelected>>", partial(combocallback, extra))
            self.combos[extra].current(len(functions))
            self.combos[extra].grid(row=row-1, column=col)

        self.expoScale = Scale(self.BottomFrame,
                               value=curr_config_data["exposition"],
                               orient=HORIZONTAL, from_=0, to=1020, length=300,
                               command=partial(scalecallback, "exposition"))
        self.expoScale.grid(row=0, column=0)
        self.expoLabel = Label(self.BottomFrame, text="Exposition")
        self.expoLabel.grid(row=1, column=0)

        self.CaptDelayScale = Scale(self.BottomFrame,
                                    value=curr_config_data["capture_delay"],
                                    orient=HORIZONTAL, from_=0, to=1020, length=300,
                                    command=partial(scalecallback, "capture_delay"))
        self.CaptDelayScale.grid(row=0, column=1, padx=10)
        self.CaptDelayLabel = Label(self.BottomFrame, text="capture_delay")
        self.CaptDelayLabel.grid(row=1, column=1)

        self.SaveButton = Button(self.BottomFrame, text="Save", command=buttoncallback)
        self.SaveButton.grid(row=0, column=2)

        if not self.refreshIMG:
            self.refreshImages()

        self.refreshGUI = False

    def refreshImages(self):

        self.refreshIMG = True

        if not self.refreshGUI:
            if not self.refreshGUI:
                for i, img in enumerate(self.images):
                    img.image = (self.pictures[i])
                    img.configure(image=self.pictures[i])

        self.root.update_idletasks()
        self.root.after(10, self.refreshImages)


    def callback_exit(self):
        self.quitFlag = True

    def run(self):
        self.root = Tk()
        self.root.protocol("WM_DELETE_WINDOW", self.callback_exit)
        self.root.title(self.title)
        self.root.geometry(self.geometry)

        self.root.mainloop()
        return