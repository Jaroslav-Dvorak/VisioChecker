from time import sleep, time
from PIL import Image, ImageTk
import GUI
import sock_comm
import configRW

configfile = "connections.ini"

AllconfigData = configRW.read_config(configfile)
RpiAvaible = {}
lastState = ""
try:
    lastState = AllconfigData["MAIN_CONFIG"]["last_state"]
    lastIP = AllconfigData["IP"][lastState]
    RpiAvaible = AllconfigData["IP"]
except KeyError:
    try:
        configRW.write_config({"IP": {"localhost": "127.0.0.1:2020"}, "MAIN_CONFIG": {"last_state": "localhost"}},
                              configfile)
        print("Konfigurační soubor neexistuje nebo není ve správném formátu.\n"
              "Vytvořen nový.")
        AllconfigData = configRW.read_config(configfile)
        lastState = AllconfigData["MAIN_CONFIG"]["last_state"]
        RpiAvaible = AllconfigData["IP"]

    except Exception as e:
        print("error:", e)
        exit()
except Exception as e:
    print("error:", e)
    exit()

ConnectSelector = GUI.ConnectSelector(rpidict=RpiAvaible, last=lastState)

if ConnectSelector.quitFlag:
    exit()

AllconfigData["IP"] = ConnectSelector.rpidict
AllconfigData["MAIN_CONFIG"]["last_state"] = ConnectSelector.laststate.lower()
configRW.write_config(AllconfigData, configfile)

ip, port = ConnectSelector.IP_PORT


LAST_MESSAGE = None
CLI = sock_comm.Client(ip=ip, port=port)


def SEND(obj):
    CLI.send_message(obj)
    return True, time()

def GET():
    last_msg = CLI.LAST_MESSAGE
    if last_msg is None:
        return None
    else:
        new_msg = CLI.LAST_MESSAGE
        CLI.LAST_MESSAGE = None
        return new_msg


NEW_MESSAGE = None
while NEW_MESSAGE is None:
    sleep(1)
    NEW_MESSAGE = GET()

AllConfigData, Visio_core, comCommands = NEW_MESSAGE
# exit()
Funcs = Visio_core()
CurrConfigData = AllConfigData[AllConfigData["MAIN_CONFIG"]["last_state"]]

requestIMGmsg = comCommands["CroppedIMG_request"]
SaveCorfim = comCommands["Save_corfim"]

# print("AllConfigData:", AllConfigData)
# print("VisioCore:", VisioCore)
# print("Funcs:", Funcs)
# print("CurrConfigData:", CurrConfigData)

window = GUI.Vizualizace(geometry='1920x920', title="Visio")

ActComm, StartComTime = SEND(requestIMGmsg)

OrigIMG = None
Start = True
while True:
    NEW_MESSAGE = GET()
    if NEW_MESSAGE is not None:
        # print("Incoming message, type:", type(NEW_MESSAGE).__name__)
        if type(NEW_MESSAGE).__name__ == "ndarray":
            print("incoming image, time after request:", time()-StartComTime)
            OrigIMG = NEW_MESSAGE
            ActComm = False

        if type(NEW_MESSAGE).__name__ == "str" and NEW_MESSAGE == SaveCorfim:
            print("config saved, time after request:", time() - StartComTime)
            ActComm = False

    if OrigIMG is not None:
        Images = []
        Images.append(OrigIMG)

        NewImages = Visio_core(input_image=OrigIMG, parameters=CurrConfigData, order=CurrConfigData["order"])
        if NewImages is not None:
            Images += NewImages
        if Start or window.refreshGUI:
            StartImages = [ImageTk.PhotoImage(image=Image.fromarray(Img).resize((300, 300))) for Img in Images]
            window.startGUI(images=StartImages, curr_config_data=CurrConfigData, functions=Funcs)
            Start = False
        else:
            if not window.refreshGUI:
                ConvImages = [ImageTk.PhotoImage(image=Image.fromarray(Img).resize((300, 300))) for Img in Images]
                try:
                    window.pictures = ConvImages
                except Exception as e:
                    print("chyba překreslení")
                if len(window.ScaleValues) > 0:
                    CurrConfigData.update(window.ScaleValues)
                    window.ScaleValues = {}
        if window.quitFlag:
            window.root.quit()
            CLI.S.close()
            break

        if window.saveCommand and not ActComm:
            ActComm, StartComTime = SEND(CurrConfigData)
            window.saveCommand = False
            ActComm = True

        if not ActComm:
            ActComm, StartComTime = SEND(requestIMGmsg)
            ActComm = True

    sleep(0.1)
