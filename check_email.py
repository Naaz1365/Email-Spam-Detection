import os

folder = os.path.expandvars(
    r"%USERPROFILE%\Downloads\enron_mail_20150507\maildir\allen-p\inbox"
)

files = os.listdir(folder)

for filename in files[:5]:

    file_path = "\\\\?\\" + os.path.join(folder, filename)

    print("\nEmail:", filename)

    try:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            print(f.read(500))

    except Exception as e:
        print("Error:", e)