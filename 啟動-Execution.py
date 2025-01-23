import os, sys, subprocess

def SystemPrint(message):
    print(f"System: {message}")
SystemPrint("Started checking.")

def ExitProgram():
    input("Press any key to exit. . .")
    exit()

def CheckEnvironment():
    #Only support python 3.10 or higher version
    current_version, min_version = sys.version_info, (3, 10, 0, 'final', 0)
    strPyVersion = str(current_version[0]) + "." + str(current_version[1]) + "." + str(current_version[2])
    SystemPrint(f"Python version: {strPyVersion}")
    if (current_version < min_version):
        SystemPrint("Please update your Python version to 3.10 or higher to execute this application.")
        ExitProgram()
    
    SystemPrint("Checking completed.")

def main():
    try:
        CheckEnvironment()
        dirname = os.path.join(os.path.dirname(__file__), "源代碼-Source_code", "AntiScamChatBot.py")
        command = f"{sys.executable} -m streamlit run \"{dirname}\""
        print(os.getcwd())
        print(command)
        input()

        subprocess.Popen(command, shell=True, stderr=subprocess.PIPE, stdout=subprocess.PIPE)
    except Exception as e:
        SystemPrint(f"Error! Reason: {e}")
        ExitProgram()

if __name__ == "__main__":
    main()