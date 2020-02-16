from time import sleep, time
from PIL import Image, ImageTk
import GUI
import sock_comm
import configRW

configfile = "connections.ini"

AllconfigIpData = configRW.read_config(configfile)
RpiAvaible = {}
lastIpState = ""
try:
    lastIpState = AllconfigIpData["MAIN_CONFIG"]["last_state"]
    lastIP = AllconfigIpData["IP"][lastIpState]
    RpiAvaible = AllconfigIpData["IP"]
except KeyError:
    try:
        configRW.write_config({"IP": {"localhost": "127.0.0.1:2020"}, "MAIN_CONFIG": {"last_state": "localhost"}},
                              configfile)
        print("Konfigurační soubor neexistuje nebo není ve správném formátu.\n"
              "Vytvořen nový.")
        AllconfigIpData = configRW.read_config(configfile)
        lastIPState = AllconfigIpData["MAIN_CONFIG"]["last_state"]
        RpiAvaible = AllconfigIpData["IP"]

    except Exception as e:
        print("error:", e)
        exit()
except Exception as e:
    print("error:", e)
    exit()

ConnectSelector = GUI.ConnectSelector(rpidict=RpiAvaible, last=lastIpState)

if ConnectSelector.quitFlag:
    exit()

AllconfigIpData["IP"] = ConnectSelector.rpidict
AllconfigIpData["MAIN_CONFIG"]["last_state"] = ConnectSelector.laststate.lower()
configRW.write_config(AllconfigIpData, configfile)

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
Funcs = Visio_core()


requestIMGmsg = comCommands["CroppedIMG_request"]
SaveCorfim = comCommands["Save_corfim"]

ActComm, StartComTime = SEND(requestIMGmsg)


def ratioresize(img, width):
    high = int(img.shape[0]*width/img.shape[1])
    img = Image.fromarray(img).resize((width, high))
    return img


Width = 300
OrigIMG = None
Start = True
window = GUI.Vizualizace(geometry='1920x920', title="Visio")
while True:
    lastState = AllConfigData["MAIN_CONFIG"]["last_state"]
    CurrConfigData = AllConfigData[lastState]

    NEW_MESSAGE = GET()
    if NEW_MESSAGE is not None:
        # print("Incoming message, type:", type(NEW_MESSAGE).__name__)
        if type(NEW_MESSAGE).__name__ == "ndarray":
            print("incoming image, time after request:", time()-StartComTime)
            OrigIMG = NEW_MESSAGE
            ActComm = False

        if type(NEW_MESSAGE).__name__ == "str" and NEW_MESSAGE == SaveCorfim:
            print("config saved, time after request:", time() - StartComTime)
            window.SaveButton.config(bg="green")
            ActComm = False

    if OrigIMG is not None:
        Images = []
        Images.append(OrigIMG)

        NewImages = Visio_core(input_image=OrigIMG, parameters=CurrConfigData, order=CurrConfigData["order"])
        if NewImages is not None:
            Images += NewImages
        if Start or window.refreshGUI:
            StartImages = [ImageTk.PhotoImage(image=ratioresize(Img, width=Width)) for Img in Images]
            window.startGUI(images=StartImages, all_config_data=AllConfigData, functions=Funcs)
            Start = False
        else:
            if not window.refreshGUI:
                ConvImages = [ImageTk.PhotoImage(image=ratioresize(Img, width=Width)) for Img in Images]
                try:
                    window.pictures = ConvImages
                except Exception as e:
                    print("chyba překreslení")
                if len(window.ScaleValues) > 0:
                    CurrConfigData.update(window.ScaleValues)
                    window.ScaleValues = {}
                if window.newpreset:

                    AllConfigData = window.newpreset
                    window.newpreset = None

                    # if window.newpreset in AllConfigData:
                    #     for items in window.images + window.scales + window.combos + window.labels:
                    #         items.destroy()
                    #         window.TopFrame.destroy()
                    #         window.MiddleFrame.destroy()
                    #     window.saved = False
                    #     window.SaveButton.config(bg="red")
                    #     window.refreshGUI = True
                    # else:
                    #     AllConfigData[window.newpreset] = AllConfigData[lastState]
                    #     window.newpreset = None
                    # AllConfigData["MAIN_CONFIG"]["last_state"] = window.newpreset

        if window.quitFlag:
            window.root.quit()
            CLI.S.close()
            break

        if window.saveCommand and not ActComm:
            ActComm, StartComTime = SEND(AllConfigData)
            window.saveCommand = False

        if not ActComm:
            ActComm, StartComTime = SEND(requestIMGmsg)

    sleep(0.1)
