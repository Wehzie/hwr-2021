"""
Redacts a .env file by removing the values but keeping the keys.
"""

with open(".env", "r") as f:
    original = f.readlines()

redacted = []
for line in original:
    if "=" in line:
        redacted.append(line.split("=", 1)[0] + "=\n")
    else:
        redacted.append(line)

with open("redacted.env", "w") as f:
    f.writelines(redacted)
