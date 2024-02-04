import keyboard
import time

temp_pose = [0.0, 0.0, 0.0]
plusminus = True

def write(txt):
    with open('change_pose.txt', 'w') as f:
        f.write('{0}, {1}, {2}'.format(txt[0],txt[1],txt[2]))
        print("\npose : {0}, plusminus : {1}".format(temp_pose, plusminus))    
        time.sleep(0.1)

write(temp_pose)        


while True:
    if keyboard.read_key() == "0":
        if plusminus:
            plusminus = False
        else:
            plusminus = True
    if plusminus:
        if keyboard.read_key() == "1":
            temp_pose[0] += 1.0
            write(temp_pose)
        if keyboard.read_key() == "2":
            temp_pose[1] += 1.0                    
            write(temp_pose)
        if keyboard.read_key() == "3":
            temp_pose[2] += 1.0
            write(temp_pose)
        if keyboard.read_key() == "4":
            temp_pose[0] += 0.1
            write(temp_pose)
        if keyboard.read_key() == "5":
            temp_pose[1] += 0.1
            write(temp_pose)
        if keyboard.read_key() == "6":
            temp_pose[2] += 0.1       
            write(temp_pose)
    else:
        if keyboard.read_key() == "1":
            temp_pose[0] -= 1.0
            write(temp_pose)
        if keyboard.read_key() == "2":
            temp_pose[1] -= 1.0                    
            write(temp_pose)
        if keyboard.read_key() == "3":
            temp_pose[2] -= 1.0
            write(temp_pose)
        if keyboard.read_key() == "4":
            temp_pose[0] -= 0.1
            write(temp_pose)
        if keyboard.read_key() == "5":
            temp_pose[1] -= 0.1
            write(temp_pose)
        if keyboard.read_key() == "6":
            temp_pose[2] -= 0.1       
            write(temp_pose)
    if keyboard.read_key() == "q":
        temp_pose = [0.0, 0.0, 0.0]
        write(temp_pose)
    time.sleep(1)


