from ultralytics import YOLO
import cv2
import os
import numpy as np
from collections import defaultdict
import math
from scipy.signal import convolve
import json
from functools import singledispatch
 

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)
print(dname)
model = YOLO('yolov8x-pose-p6.pt')  # load an official model

##EXTERNAL INPUTS#
person_height = 180
selected_fps=0
video_side=1 #1 left, 2 right
video_id="demo"
##EXTERNAL INPUTS



cap = cv2.VideoCapture("original_videos/"+video_id+".mp4")
if(selected_fps==0):
    fps = cap.get(cv2.CAP_PROP_FPS)
else:
    fps=selected_fps

vwidth = cap.get(cv2.CAP_PROP_FRAME_WIDTH )
vheight = cap.get(cv2.CAP_PROP_FRAME_HEIGHT )
VIDEO_HEIGHT = 640
FLOOR_SECURE_DISTANCE=5
r = VIDEO_HEIGHT/ vheight
vwidth = int(vwidth*r)
vheight=VIDEO_HEIGHT

#cap = cv2.VideoCapture("test.jpg")
track_history = defaultdict(lambda: [])
hip_history_front =[]
hip_history_back =[]
previous_keypoint=[]
lean_angles=[]

first_contact_angles=[]
contact_angles=[]
toeoff_angles=[]

vertical_displacement=[]
vertical_displacement_from_floor=[]
contact_times=[]
flying_times=[]

max_distance_legs=0
max_height = 0
startpoint = 0 
endpoint = 0
avg_speed = 0
previous_direction = 0
previos_hip_distance = 0
frames=0

contact_time=0
flying_time=0

step_counter = 0
floor_position = 0
stage = "LEARNING"
stage_color = (66,66,66)
trigger_first_contact=False


@singledispatch
def to_serializable(val):
    """Used by default."""
    return str(val)


@to_serializable.register(np.float32)
def ts_float32(val):
    """Used if *val* is an instance of numpy.float32."""
    return np.float64(val)

def calculate_distance(leg1, previous_keypoint):
   dx = leg1[0]-previous_keypoint[0]
   dy = leg1[1]-previous_keypoint[1]
   distance = math.sqrt(dx**2+dy**2)
   return distance
def calculate_distance_x(leg1, previous_keypoint):
   dx = leg1[0]-previous_keypoint[0]
   return dx

def getAngle(a, b, c):
    ang = math.degrees(math.atan2(c[1]-b[1], c[0]-b[0]) - math.atan2(a[1]-b[1], a[0]-b[0]))
    return ang + 360 if ang < 0 else ang

fourcc = cv2.VideoWriter_fourcc('H','2','6','4')
out = cv2.VideoWriter("\\test_results\\"+video_id+'_analysis.mp4',fourcc, fps, (vwidth,vheight))


while cap.isOpened():
    #input("Press Enter to continue...")
    ret, frame = cap.read()

    if ret == False:
            break
    
          # Run YOLOv8 inference on the frame
    results = model.predict(frame,conf=0.8,imgsz=640,show_labels=False,show_conf=False,line_width=0)

        # Visualize the results on the frame
    #annotated_frame = results[0].plot()
    annotated_frame = frame

    #result_keypoint = results[0].leg1.xyn.cpu().numpy()
    height, width, _ = annotated_frame.shape

    keypoints = results[0].keypoints.xy.cpu().numpy()[0]
    box = results[0].boxes.xywh.cpu().numpy()
    if(len(box)>0):
         box=box[0]

         if(max_height<box[3]):
             max_height=box[3]

  
    if len(keypoints)>=16:
        proportion = person_height/max_height




        if(video_side==1):

            nose = keypoints[0]
            leg1 = keypoints[15]
            leg2 = keypoints[16]

            knee1 = keypoints[13]
            knee2 = keypoints[14]

            hip1 = keypoints[11]
            hip2 = keypoints[12]

            shoulder1=keypoints[5]
            shoulder2=keypoints[6]


        else:
            nose = keypoints[0]
            leg1 = keypoints[16]
            leg2 = keypoints[15]

            knee1 = keypoints[14]
            knee2 = keypoints[13]

            hip1 = keypoints[12] 
            hip2 = keypoints[11]

            shoulder1=keypoints[6]
            shoulder2=keypoints[5]

        hipcenter=((hip1[0]+hip2[0])/2,(hip1[1]+hip2[1])/2)
        torsocenter=((shoulder1[0]+shoulder2[0])/2,(shoulder1[1]+shoulder2[1])/2)

        hipProyection = (hipcenter[0],nose[1])

        if(len(previous_keypoint)==0):
             previous_keypoint = leg1

        #foot error correction
        if(abs(leg1[1]-previous_keypoint[1])>abs(leg2[1]-previous_keypoint[1]) and abs(leg1[0]-leg2[0])*proportion<25):
             leg1 = leg2
             leg2 = leg1

        leg_distance = calculate_distance_x(leg1,leg2)

        if(abs(leg_distance*proportion)>max_distance_legs and stage != "LEARNING"):
            max_distance_legs = abs(leg_distance)*proportion
            

        direction = calculate_distance_x(leg1,previous_keypoint)
        direction_force = direction/abs(direction)
        instant_speed = abs(direction)*proportion*fps
        
        m_per_s = instant_speed / 100
        km_per_min = m_per_s * 60 / 1000
        min_per_km = 1 / km_per_min
        min_per_km=min_per_km/2

        if(min_per_km>0 and min_per_km<30):
             frames+=1
             if(avg_speed==0):
                avg_speed = min_per_km
             else:
                  avg_speed = ((avg_speed*(frames-1))+min_per_km)/(frames)
                  #avg_speed = (avg_speed+min_per_km)/2

        if(previous_direction==0):
             previous_direction = direction_force
      
        #print(str(min_per_km)+"min/km")
        #print(str(avg_speed)+"min/km")

        legangle= getAngle(hip1,knee1,leg1)
        if(legangle>180):
            legangle=360-legangle
        vertical_displacement.append(hip1[1])

        if(stage != "LEARNING"):
                vertical_displacement_from_floor.append((floor_position-hip1[1])*proportion)

        if(step_counter<4):
             if(leg1[1]>floor_position):
                  floor_position=leg1[1]                
        else:
             
             if(leg1[1]>=floor_position-(FLOOR_SECURE_DISTANCE/proportion)):
                  if(stage=="SWING"):
                    stage="1ST CONTACT"
                    contact_time=0
                    flying_times.append(round((flying_time/fps)*1000,0))
                    flying_time=0
                    stage_color =(0,0,255)
                    trigger_first_contact=True
                  else:
                    if(trigger_first_contact):
                        first_contact_angles.append(legangle)
                        trigger_first_contact=False
                    stage="CONTACT"
                    contact_time+=1
                    stage_color =(157,0,219)
             else:
                  if(stage=="CONTACT"):
                    stage="HEEL OFF"
                    contact_times.append(round((contact_time/fps)*1000,0))
                    contact_time=0
                    flying_time=0
                    toeoff_angles.append(legangle)
                    stage_color =(219,160,0)
                  else:
                      stage="SWING"
                      flying_time+=1
                      stage_color =(100,163,0)

        cv2.line(annotated_frame, (int(0), int(floor_position-(FLOOR_SECURE_DISTANCE/proportion))), (int(width), int(floor_position-(FLOOR_SECURE_DISTANCE/proportion))),  (255, 255, 255), 1)
        #print(stage)

        min_vertical_displacement=np.min(vertical_displacement)
        max_vertical_displacement=np.max(vertical_displacement)
        cv2.line(annotated_frame, (int(0), int(max_vertical_displacement)), (int(width), int(max_vertical_displacement)),  (255, 0, 0), 1)
        cv2.line(annotated_frame, (int(0), int(min_vertical_displacement)), (int(width), int(min_vertical_displacement)),  (255, 0, 0), 1)

        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(annotated_frame, ""+str(round(abs(max_vertical_displacement-min_vertical_displacement)*proportion,1))+"cm", (0,int(min_vertical_displacement)+25 ), font, 1, 
                 (255, 0, 0), 1, cv2.LINE_AA, False)


        hipdistance = calculate_distance_x(hip1,leg1)*proportion

        if(previos_hip_distance == 0):
             previos_hip_distance = hipdistance

        if(previous_direction!=direction_force and abs(direction)*proportion > 3):#cambio de direccion
            step_counter+=1
            if(video_side==2):
                if(hipdistance>0):
                    hip_history_back.append(previos_hip_distance)
                else:
                    hip_history_front.append(previos_hip_distance)

            else:
                if(hipdistance>0):
                    hip_history_front.append(previos_hip_distance)
                else:
                    hip_history_back.append(previos_hip_distance)
            
        '''''
        print("DIRECTION:")
        print(direction_force)
        print(direction*proportion)

        print("STEPS:")
        print(step_counter)
        
        
        print("HIP:")
        print(np.mean(hip_history_front))
        print(np.mean(hip_history_back))
        #print(np.min(hip_history))



        
        #print(direction)
        #print("leg" + str(leg_distance))

       # print(max_height)
       

       # print(leg_distance*proportion)
        
        '''
        
        cv2.circle(annotated_frame, (int(leg1[0]), int(leg1[1])), 5, stage_color, thickness=-1, lineType=cv2.FILLED)
        cv2.circle(annotated_frame, (int(leg1[0]), int(leg1[1])), 15, stage_color, 5)

        cv2.circle(annotated_frame, (int(leg2[0]), int(leg2[1])), 5, (255, 255, 255), thickness=-1, lineType=cv2.FILLED)
        cv2.circle(annotated_frame, (int(leg2[0]), int(leg2[1])), 15,  (255, 255, 255), 5)

        cv2.circle(annotated_frame, (int(hip1[0]), int(hip1[1])), 5, (255, 255, 255), thickness=-1, lineType=cv2.FILLED)
        cv2.circle(annotated_frame, (int(hip1[0]), int(hip1[1])), 15,  (255, 255, 255), 5)
        
        cv2.line(annotated_frame,  (int(hip1[0]), int(hip1[1])),(int(knee1[0]), int(knee1[1])), (255, 255, 255), 2)
        cv2.line(annotated_frame, (int(knee1[0]), int(knee1[1])), (int(leg1[0]), int(leg1[1])),  (255, 255, 255), 2)

        cv2.line(annotated_frame, (int(hip1[0]), int(hip1[1])), (int(hipProyection[0]), int(hipProyection[1])),  (255, 255, 255), 1)
        cv2.line(annotated_frame, (int(hip1[0]), int(hip1[1])), (int(torsocenter[0]), int(torsocenter[1])),  (255, 255, 255), 1)

        lean_angle= getAngle(torsocenter,hipcenter,hipProyection)
        if(lean_angle>90):
            lean_angle=360-lean_angle
        
        #print(lean_angle)
        if(lean_angle<90 and stage=="CONTACT"):
            lean_angles.append(lean_angle)

        
        

        #print(angle)
        
        ## TRACKING LEG ##
        track = track_history[0]
        track.append((float(leg1[0]), float(leg1[1])))  # x, y center point
        if len(track) > 20:  # retain 90 tracks for 90 frames
                track.pop(0)

        # Draw the tracking lines
        points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
        cv2.polylines(annotated_frame, [points], isClosed=False, color=(0, 0, 255), thickness=3)
        ## END TRACKING ###

        previous_keypoint = leg1   
        if( abs(direction)*proportion > 3):
            previous_direction = direction_force

        previos_hip_distance = hipdistance

    r = VIDEO_HEIGHT / height
    dim = (int(width * r), VIDEO_HEIGHT)
    annotated_frame = cv2.resize(annotated_frame,dim,interpolation=cv2.INTER_AREA)

    text_speed = str(round(avg_speed, 1))+" min/km"

    mean_angles = np.mean(lean_angles)
    if(math.isnan(mean_angles)):
         mean_angles=0
    text_leanangle=str(round(mean_angles,1))
    
    font = cv2.FONT_HERSHEY_SIMPLEX

    mean_front = np.mean(hip_history_front)
    if(math.isnan(mean_front)):
         mean_front=0

    mean_back = np.mean(hip_history_back)
    if(math.isnan(mean_back)):
         mean_back=0


    text_hip_front = str(round(abs(mean_front),1))+"cm"
    text_hip_back = str(round(abs(mean_back),1))+"cm"

    vpos=0
    twidth=170
    cv2.rectangle(annotated_frame,(vwidth-twidth,vpos), (vwidth-2,vpos+30), (0,0,255), -1)
    cv2.putText(annotated_frame, "PACE:"+text_speed, (vwidth-twidth+10,vpos+20), font, 1/2, 
                 (255,255,255), 1, cv2.LINE_AA, False)
    vpos+=35
    twidth=90
    cv2.rectangle(annotated_frame,(vwidth-twidth,vpos), (vwidth-2,vpos+30), (0,0,255), -1)
    cv2.putText(annotated_frame, "STEPS:"+str(step_counter), (vwidth-twidth+10,vpos+20), font, 1/2, 
                 (255,255,255), 1, cv2.LINE_AA, False)
    
    vpos+=35
    twidth=80
    cv2.rectangle(annotated_frame,(vwidth-twidth,vpos), (vwidth-2,vpos+30), (0,0,255), -1)
    cv2.putText(annotated_frame, "LEAN: +"+text_leanangle, (vwidth-twidth+10,vpos+20), font, 1/3, 
                 (255,255,255), 1, cv2.LINE_AA, False)
  
    vpos+=35
    twidth=185
    cv2.rectangle(annotated_frame,(vwidth-twidth,vpos), (vwidth-2,vpos+30), (0,0,255), -1)
    cv2.putText(annotated_frame, "STRIDING: +"+text_hip_front+" / -"+text_hip_back, (vwidth-twidth+10,vpos+20), font, 1/3, 
                 (255,255,255), 1, cv2.LINE_AA, False)


    vpos=VIDEO_HEIGHT-50
    twidth=120
    cv2.rectangle(annotated_frame,(int((dim[0]/2)-(twidth/2)),vpos), (int((dim[0]/2)+(twidth/2)),vpos+30), stage_color, -1)
    cv2.putText(annotated_frame, stage, (int((dim[0]/2)-(twidth/2))+10,vpos+20), font, 1/2, 
                 (255,255,255), 1, cv2.LINE_AA, False)
                 

    if(stage=="1ST CONTACT"):
        cv2.imwrite(video_id+"_1stcontact.jpg", annotated_frame)

    if(stage=="HEEL OFF"):
        cv2.imwrite(video_id+"_heeloff.jpg", annotated_frame)

    out.write(annotated_frame)
    out.write(annotated_frame)
    out.write(annotated_frame)
        # Display the annotated frame
    cv2.imshow("YOLOv8 Inference", annotated_frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

analysis = {
    "lean_angles": lean_angles,
    "hip_history_front":hip_history_front,
    "hip_history_back":hip_history_back,
    "lean_angles":lean_angles,
    "first_contact_angles":first_contact_angles,
    "toeoff_angles":toeoff_angles,
    "vertical_displacement_from_floor":vertical_displacement_from_floor,
    "max_distance_legs":max_distance_legs,
    "flying_times":flying_times,
    "contact_times":contact_times
}

print(analysis)
 
# Serializing json
json_object = json.dumps(analysis, indent=4, default=to_serializable)
 
# Writing to sample.json
with open(video_id+".json", "w") as outfile:
    outfile.write(json_object)


cap.release()
out.release()
#cv2.destroyAllWindows()
