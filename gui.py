from tkinter import *
from PIL import ImageTk, Image
import shutil
import random
from tkinter import messagebox

import argparse
import scipy.io
import torch
import numpy as np
import os
from torchvision import datasets

# Evaluate
parser = argparse.ArgumentParser(description='Demo')
parser.add_argument('--test_dir', default=r'put_cameras\obiekty_sportowe', type=str, help='./test_data')
parser.add_argument('--query_cam_index', default=1, type=int, help='which camera will be query (the rest of cameras will be gallery)')
opts = parser.parse_args()

# ------------VARIABLES----------------
images = []  # list with path to selected images

query_cam_index = opts.query_cam_index
query_cam_index = int(query_cam_index)

test_dir = opts.test_dir
data_dir = os.path.join(os.getcwd(), test_dir + '/pytorch')
dirs = os.listdir(data_dir)
# initializing image dataset
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x)) for x in dirs}

repetitions = {}
query_path = ''

query_name = ""

# loading results from test_put.py
filename = 'pytorch_result.mat'
average_filename = 'pytorch_result_averages.mat'
result = scipy.io.loadmat(test_dir + "/" + filename)
avg_result = scipy.io.loadmat(test_dir + "/" + average_filename)

features = torch.FloatTensor(result['feature'])
cams = result['cam'][0]
labels = result['label'][0]
avg_features = torch.FloatTensor(avg_result['feature'])
avg_cams = avg_result['cam'][0]
avg_labels = avg_result['label'][0]



# initializing array with names of directories for future references
dir_inds = []
for subdir in dirs:
    dir_inds.append(str(subdir))

cameras = np.unique(cams)
gallery_lengths = []

itr = 1
for cam in cameras:
    if itr == query_cam_index:
        qf_inds = cams == cam
        query_features = features[qf_inds]
        query_features = query_features.cpu()
        query_labels = labels[qf_inds]
        query_cams = cams[qf_inds]
        gf_inds = cams != cam
        gallery_features = features[gf_inds]
        gallery_features = gallery_features.cpu()
        gallery_labels = labels[gf_inds]
        gallery_cams = cams[gf_inds]

        # average values
        qf_inds = avg_cams == cam
        avg_query_features = avg_features[qf_inds]
        avg_query_features = avg_query_features.cpu()
        avg_query_labels = avg_labels[qf_inds]
        avg_query_cams = avg_cams[qf_inds]

        gf_inds = avg_cams != cam
        avg_gallery_features = avg_features[gf_inds]
        avg_gallery_features = avg_gallery_features.cpu()
        avg_gallery_labels = avg_labels[gf_inds]
        avg_gallery_cams = avg_cams[gf_inds]

        gallery_scores = np.zeros(len(avg_gallery_features))
    else:
        gallery_len = len(np.argwhere(cams == cam))
        gallery_lengths.append(gallery_len)
        repetitions[int(cam)] = []
    itr += 1

j = 2  # row indicator
k = 0  # column indicator


# ------------FUNCTIONS----------------

# used to determine in which folder is a photo and what is its index in that folder
def calculateLen(lengths, index):
    bound = index
    # if there are more than one match in index array
    if isinstance(index, np.ndarray):
        if index[0] < lengths[0]:
            return index
        else:
            bound = index[0]

    elif index < lengths[0]:
        return index
    result = 0
    i = 0

    while result < bound:
        result += lengths[i]
        i += 1
    return index - (result - lengths[i - 1])

def calculate_scores(index, bound):
    global gallery_scores
    for i in range(bound):
        cam_ind = int(gallery_cams[index[i]])
        label = int(gallery_labels[index[i]])
        mask1 = np.argwhere(avg_gallery_labels==label)
        mask2 = np.argwhere(avg_gallery_cams==cam_ind)
        ind = np.intersect1d(mask1, mask2)[0]
        gallery_scores[ind] += 1
    sorted_index = np.argsort(gallery_scores)
    # from max to min
    return sorted_index[::-1]

# checking if input is a number or if number isn't greater than features size - otherwise message pop up
def checkInputField():
    input = input_number.get()
    if (input.isnumeric()):  # only numbers accepted
        nr = int(input)
        # calculating upper bound
        query_len = len(query_labels) - 1
        if (int(r.get()) == 2):  # searching for averages
            # calculating upper bound for averages
            query_mean_len = len(avg_query_labels) - 1
            if ((nr < 0) or (nr > query_mean_len)):  # out of bounds
                messagebox.showinfo("Alert", "Given number must be between 0 and " + str(query_mean_len) + ".")
                return False
            else:
                return True
        elif ((nr < 0) or (nr > query_len)):  # out of bounds
            messagebox.showinfo("Alert", "Given number must be between 0 and " + str(query_len) + ".")
            return False
        else:
            return True
    else:  # not a number
        messagebox.showinfo("Alert", "You must use only numbers")
        return False


# clear frames
def clearFrames():
    for widget in main_frame.winfo_children():
        widget.destroy()

    for widget in photos_frame.winfo_children():
        widget.destroy()


# copying selected photos and placing them in 'output_gallery' folder
def confirmChoice():
    global images, query_path, query_name
    counter = 0
    database_dir = os.path.join(os.getcwd(), 'output_gallery')
    # dir_test = r'D:\Studia\sem7\Person_reID_baseline_pytorch\output_gallery'
    try:
        os.mkdir(database_dir)  # making main folder for subfolders for the objects
    except Exception as e:
        print("The folder already exists")
    dirs_test = os.listdir(database_dir)
    for subdir in dirs_test:
        counter += 1
    output_dir = database_dir + '/' + str(counter)
    os.mkdir(output_dir)  # making folder for photos of the object

    nowy = os.path.basename(query_path)  # getting query object path
    nowy_path = query_path.replace(nowy, "")  # getting query object folder path
    with os.scandir(nowy_path) as entries:
        for entry in entries:
            shutil.copy(entry, output_dir)  # copying query photos to new object folder

    for i in images:
        nowy = os.path.basename(i)  # getting candidate object path
        nowy_path = i.replace(nowy, "")  # getting candidate object folder path
        with os.scandir(nowy_path) as entries:
            for entry in entries:
                shutil.copy(entry, output_dir)  # copying candidates photos to new object folder
    print("Person", query_name, "added to database")
    images.clear()


# creating buttons with pictures of candidates
def createCandidatePicture(i, img_path, label, cam_ind):
    global j, k

    # creating buttons with pictures of candidates
    my_pic = Image.open(img_path)
    resized = my_pic.resize((100, 220), Image.ANTIALIAS)
    new_pic = ImageTk.PhotoImage(resized)

    btn = Button(photos_frame, image=new_pic, width=160, relief=FLAT)
    btn.config(command=lambda m=btn, n=img_path: [selectButton(m, n)])
    btn.image = new_pic  # keeping a reference
    btn.grid(row=j, column=k, padx=5, pady=5)

    # creating labels for photos
    if (r.get() == 2):
        button_label = "Top " + str(i + 1) + "\nperson id: " + str(label) + "\nCam " + str(cam_ind)
    else:
        button_label = "Person " + str(int(label)) + "\nCam " + str(int(cam_ind))
    photo_label = Label(photos_frame, text=button_label, font=25)
    photo_label.grid(row=j - 1, column=k)
    k += 1
    if (k == 5):
        j = 4
        k = 0


# creating text on the top of the window
def createLabel():
    # ---------MAIN LABEL----------
    mainLabel = Label(root, text="CHOOSE THE SAME PEOPLE", font=("Arial", 25))
    mainLabel.grid(row=0, columnspan=15)


# creating main picture
def createMainPicture(i, query_cam_index, query_path):
    # creating label for main object
    global query_name
    query_name = i
    main_pic_label_text = "Person " + str(query_name) + "\nCam " + str(query_cam_index)
    main_pic_label2 = Label(main_frame, text=main_pic_label_text)
    main_pic_label2.grid(row=0, column=0)

    # creating picture of main object
    main_pic = Image.open(query_path)
    resized_main_pic = main_pic.resize((120, 240), Image.ANTIALIAS)
    new_main_pic = ImageTk.PhotoImage(resized_main_pic)
    main_pic_label = Label(main_frame, image=new_main_pic)
    main_pic_label.grid(row=1, column=0)

    main_pic_label.image = new_main_pic  # keeping a reference


# disabling checkbox after choosing option "averages"
def disableCheckbox():
    if (int(r.get()) == 2 or int(r.get()) == 3):
        duplicate_checkbox.config(state="disabled")
        duplicate_checkbox.deselect()
    else:
        duplicate_checkbox.config(state="normal")


# BIG FUNKCJA
def ranking():
    global j, k, query_path, gallery_scores
    j = 2  # row indicator
    k = 0  # column indicator

    clearFrames()

    input_index = int(input_number.get())
    index = sortImg(query_features[input_index], gallery_features)
    if (int(r.get()) == 1 or int(r.get())==3):
        query_path, _ = image_datasets[dir_inds[query_cam_index - 1]].imgs[input_index]
        label = int(query_labels[input_index])
        createMainPicture(label, query_cam_index, query_path)
    elif (int(r.get()) == 2):
        j_var = int(input_index)
        averages_index = sortImg(avg_query_features[j_var], avg_gallery_features)
        avg_query_index = int(avg_query_labels[j_var])

        query_inds = np.where(query_labels == avg_query_index)[0]
        query_path, _ = image_datasets[dir_inds[query_cam_index - 1]].imgs[query_inds[0]]
        createMainPicture(avg_query_index, query_cam_index, query_path)
    ########################################################################
    # Visualize the rank result

    print("Query path: ", query_path)
    print('Top 10 images are as follow:')
    try:  # Visualize Ranking Result

        # Scores ranking results --------
        if (r.get() == 3):
            #print("QUERY INDEX: ", avg_query_index)
            bound = 50
            sorted_scores = calculate_scores(index, bound)
            gallery_scores = np.zeros(len(avg_gallery_features))
            for i in range(10):
                person_id = int(avg_gallery_labels[sorted_scores[i]])
                label = int(person_id)
                cam_ind = int(avg_gallery_cams[sorted_scores[i]])
                print("PERSON", i + 1, "INDEX:", person_id, "CAM:", cam_ind)
                mask1 = np.argwhere(gallery_labels == person_id)
                mask2 = np.argwhere(gallery_cams == cam_ind)
                inds = np.intersect1d(mask1, mask2)
                ind = calculateLen(gallery_lengths, inds)
                mid = int((ind[0]+ind[-1])/2)
                img_path, _ = image_datasets[dir_inds[int(cam_ind) - 1]].imgs[mid]

                createCandidatePicture(i, img_path, label, cam_ind)

                print(img_path)
        # Average ranking results ----------
        elif (r.get() == 2):
            print("AVERAGE QUERY INDEX: ", avg_query_index)

            for i in range(10):  # pierwsze zdjecie tylko
                person_id = int(avg_gallery_labels[averages_index[i]])
                label = int(person_id)
                cam_ind = int(avg_gallery_cams[averages_index[i]])
                print("PERSON", i + 1, "INDEX:", person_id, "CAM:", cam_ind)
                mask1 = np.argwhere(gallery_labels == person_id)
                mask2 = np.argwhere(gallery_cams == cam_ind)
                inds = np.intersect1d(mask1, mask2)
                ind = calculateLen(gallery_lengths, inds)
                mid = int((ind[0]+ind[-1])/2)
                img_path, _ = image_datasets[dir_inds[int(cam_ind) - 1]].imgs[mid]

                createCandidatePicture(i, img_path, label, cam_ind)

                print(img_path)

        elif (int(var.get()) == 1):
            top_ten = 0
            i = 0

            while top_ten < 10:

                cam_ind = int(gallery_cams[index[i]])

                ind = calculateLen(gallery_lengths, index[i])
                img_path, _ = image_datasets[dir_inds[cam_ind - 1]].imgs[ind]

                label = gallery_labels[index[i]]
                if not int(label) in repetitions[int(cam_ind)]:
                    createCandidatePicture(i, img_path, label, cam_ind)

                    print(img_path)
                    repetitions[int(cam_ind)].append(int(label))
                    top_ten += 1
                i += 1
            print("Avoided repetitions: ")
            for key in repetitions:
                print("Cam " + str(key) + ": person ids: " + str(repetitions[key]))
            for cam in cameras:
                repetitions[int(cam)] = []

        else:
            for i in range(10):
                cam_ind = int(gallery_cams[index[i]])
                ind = calculateLen(gallery_lengths, index[i])
                img_path, _ = image_datasets[dir_inds[cam_ind - 1]].imgs[ind]
                label = gallery_labels[index[i]]

                createCandidatePicture(i, img_path, label, cam_ind)
                print(img_path)

    except Exception as e:
        print(str(e))


# this function is called when "New Person" button is pressed
def newPerson():
    if (checkInputField()):
        createLabel()
        ranking()
        button_confirm.config(state=NORMAL)


# changing the color of the buttons
def selectButton(btn, path):
    global images
    btn.config(bg='SystemButtonFace' if btn[
                                            "background"] == "#4169e1" else "#4169e1")  # SystemButtonFace - default color, #4169e1 - royal blue
    if (btn["background"] == "#4169e1"):
        images.append(str(path))
    else:
        images.remove(str(path))


# used to determine the best match in gallery for the given query picture
def sortImg(qf, gf):
    query = qf.view(-1, 1)
    score = torch.mm(gf, query)
    score = score.squeeze(1).cpu()
    score = score.numpy()
    index = np.argsort(score)  # from small to large
    index = index[::-1]

    return index


# ---------WINDOW---------------
# creating window
root = Tk()
root.resizable(False, False)
root.title('Object detection')
root.iconbitmap('eye.ico')

# ---------FRAME---------------
# frame for main object
main_frame = LabelFrame(root, text="Main object", font=30, width=200, height=200)
main_frame.grid(row=1, column=0, padx=5, pady=5)

# frame for candidates
photos_frame = Frame(root, relief=FLAT, width=400, height=400)
photos_frame.grid(row=1, column=1, columnspan=2, rowspan=2)

# frame for options
options_frame = LabelFrame(root, text="Options", font=30)
options_frame.grid(row=2, column=0)

# frame for options
buttons_frame = Frame(root)
buttons_frame.grid(row=3, column=0)

# ------------OPTIONS------------------

# input field for main object number
input_number_label = Label(options_frame, text="person id: ", font=("Lucida", 10))
input_number_label.grid(row=0, column=0)
input_number = Entry(options_frame, justify=CENTER, width=20, font=("Lucida", 10))
input_number.grid(row=0, column=1, columnspan=2)

# no duplicates checkbox
var = IntVar()
duplicate_checkbox = Checkbutton(options_frame, text="No duplicates", variable=var, font=("Lucida", 10))
duplicate_checkbox.grid(row=1, columnspan=3)

# radiobuttons for option if average
r = IntVar()
r.set("1")
no_avg_button = Radiobutton(options_frame, text="standard", variable=r, value=1, font=("Lucida", 10),
                            command=disableCheckbox)
avg_button = Radiobutton(options_frame, text="averages", variable=r, value=2, font=("Lucida", 10),
                         command=disableCheckbox)
scores_button = Radiobutton(options_frame, text="scores", variable=r, value=3, font=("Lucida", 10),
                            command=disableCheckbox)

no_avg_button.grid(row=2, column=0)
avg_button.grid(row=2, column=1)
scores_button.grid(row=2, column=2)
# ------------BUTTONS------------------
button_new = Button(buttons_frame, text="New person", font=("Lucida", 10), command=newPerson)
button_confirm = Button(buttons_frame, text="Confirm", font=("Lucida", 10), state=DISABLED, command=confirmChoice)
button_exit = Button(buttons_frame, text="EXIT", font=("Lucida", 10), command=root.quit)

button_new.grid(row=0, column=0, padx=2)
button_confirm.grid(row=0, column=1, padx=2)
button_exit.grid(row=0, column=2, padx=2)

# ------------MAIN LOOP---------------
root.mainloop()