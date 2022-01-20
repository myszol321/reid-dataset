import os
import cv2


def crop_frame(frame, directory, tracker, frame_num, camera_index):
    for track in tracker.tracks:
        bbox = track.to_tlbr()
        class_name = track.get_class()

        x_min, y_min, x_max, y_max = bbox

        # checking if bbox is out of frame
        if x_min < 0 or y_min < 0 or y_max > frame.shape[0] or x_max > frame.shape[1]:
            continue
        
        # crop bounding-box
        cropped_frame = frame[int(y_min-5):int(y_max+5), int(x_min-5):int(x_max+5)]

        # formatting name
        person_id = str(track.track_id)
        zeros = 4 - len(person_id)
        person_id = zeros * '0' + person_id
        zeros = 6 - len(str(frame_num))
        frame_id = zeros*'0' + str(frame_num)
        img_name = person_id + '_' + str(camera_index) + '_' + frame_id + '.jpg'
        
        # determining final path to save the file
        final_path = os.path.join(directory, img_name)
        
        # making sure image is cropped properly and saving it to file
        if cropped_frame.size:
            print("Saving cropped detection...")
            print(final_path)
            print(cv2.imwrite(final_path, cropped_frame))
