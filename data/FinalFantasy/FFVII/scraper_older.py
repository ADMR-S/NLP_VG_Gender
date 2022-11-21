from urllib.request import urlopen
import time
from os import path

pages = ['Presentiment_-_AVALANCHE',
'Escape_From_Mako_Reactor_7',
'Flower_Girl_-_A_Daring_Leap',
'The_Long_Ride_Home',
'The_Reactor_Tower',
'The_Morning_After_-_Sector_7',
'The_Train_-_A_Narrow_Escape',
'Reactor_%235_-_President_Shinra',
'Flowers_Blooming_In_The_Church',
'Meet_Reno',
'Flee_The_Church',
"Let's_Go,_Bodyguard!",
'Sneak_Out_At_Midnight',
'Playground_Ai',
'Miss_Cloud%253F',
'Rescue_Tifa_From_Don_Of_The_Slums',
'Lounge_Chaos',
'Ecchi_Don',
"I'll_Smash_Them",
'Shinra_Plans',
'Encounter_At_The_Reactor_Tower',
'Sector_7_Falls',
'Biggs%252E%252E%252E_Wedge%252E%252E%252E_Jessie%252E%252E%252E_%252E%252E%252EMarlene%252E%252E%252E',
'The_Flower_Girl_-_Origins',
'A_Golden_Shiny_Wire_Of_Hope',
'140_Flights%252E%252E%252E',
'Climbing_Shinra_Tower',
'Specimen_-_Jenova',
'Experimental_Aerith',
'Captured_-_Meet_The_President',
"Sephiroth_Alive!%253F_-_Rufus'_Ascent",
'A_Confrontation_With_The_New_President',
'A_Spectacular_Escape_-_Highway_Battle',
'Ahead_On_Our_Way',
'Cloud_Begins_His_Story',
'Our_Monster',
'Nibelheim_-_Homecoming',
'The_Mountains_Of_Nibel',
'Monsters_Of_The_Nibel_Reactor',
"Sephiroth's_Thirst_For_Knowledge",
'Realization',
"Nibelheim_Aflame_-_Tifa's_Revenge",
'J-E-N-O-V-A',
'What_A_Fascinating_Story%252E%252E%252E',
'The_Village_Of_Kalm',
'Chocobo_Chocobo',
'Zolom_Impaled_-_The_Mithril_Mine',
'Junon_Fortress_-_Resuscitation',
"Rufus'_Homecoming",
'The_Boat_Ride_-_Jenova_BIRTH',
'Landing_At_Costa_Del_Sol',
'Hojo_On_The_Beach',
'Mountain_Pass_-_Another_Reactor',
"North_Corel_-_Barret's_Hometown",
'Gold_Saucer_-_Cait_Sith,_The_God_Of_Luck',
"Barret's_Crime",
'Corel_Prison',
'Gun_Arm',
'Dyne_-_Broken_Trust',
'Ester,_Your_New_Manager',
'Chocobo_Race_-_Free',
'The_Turks_-_Ruined_Reactor_-_Gongaga_Village_-_Aerith_and_Zack',
'Cosmo_Canyon_-_Grudge',
"Bugenhagen's_Observatory",
'The_Lifestream',
'Cosmo_Candle',
'Nanaki,_Son_Of_Seto',
'Red_Truth',
'Nibelheim_Revisited_-_The_Black_Cloak',
'Calamity_From_the_Skies',
'Rocket_Town_-_Hey,_Cid!',
'Sending_a_Dream_Into_the_Universe_-_Tiny_Bronco_Escapes!',
'Tiny_Bronco_Afloat',
'The_Keystone',
'Ghost_Hotel_Conversation',
'A_Midnight_Date_-_Interrupted_by_Fireworks',
'A_Spy!%253F',
'Meteor_-_A_Wound_So_Large',
'The_Black_Materia',
'Believe_In_Me_-_Cait_Sith',
'Cait_Sith_In_the_Shrinking_Temple',
"Cloud's_Sin",
"Aerith's_Wood",
"Cloud's_Self",
'Bone_Village',
'The_Sleeping_Forest_-_City_Of_the_Ancients',
"Aerith's_Gift",
'Puppet_-_Descent_Into_Life',
'Who_Am_I%252E%252E%252E%253F',
'Winter_Wonderland',
"Gast's_Videos",
'Elena_-_A_Snowy_Ride_-_Great_Glacier',
"Gaea's_Cliff_-_Mako_Crater",
'The_Shinra_Airship',
'Reunion_-_The_Black_Materia',
"Cloud's_Truth,_Or_Cloud's_Lie%253F",
'Materia_Tree',
"Sephiroth's_Ruse",
'Identification_Number_-_Out_Of_Our_Grasp_Once_More_-_Weapon_Rises',
'Origins_-_Sector_7_Train_Station',
'Dark_Sun',
'Public_Execution_-_Weapon_Attacks_-_Camaderie_-_Cheating_Death',
'Highwind_Takes_To_the_Skies',
'Mideel_-_Mind_of_a_Friend',
"Shinra's_Plans_-_Cid's_Election",
"Corel's_Huge_Materia",
"Fort_Condor's_Huge_Materia",
'Weapon_Attacks_-_Lifestream_Revisited',
'Into_Darkness',
"Enter_Cloud's_Mind_-_Nibelheim,_Long_Ago",
'That_Childhood_Promise%252E%252E%252E',
'Jealousy_-_In_My_Room',
'On_The_Other_Side_Of_the_Mountain',
'The_Truth_-_You_Were_Watching_Me_-_A_Battle_of_Will',
'Return_To_the_Others',
'Awakening',
"There_Ain't_No_Gettin'_Off",
'Underwater_Reactor_-_Sub_Space',
'Gelnika_Takes_Off_-_Countdown_to_Dreams_and_Death',
"Crash_Course_-_Shinra_No%252E_26's_Huge_Materia",
'Wreckage',
'Escape_Pod_-_Too_Little,_Too_Late',
'The_Cry_Of_the_Planet_-_That_Someone_is_Us',
'Remembrance',
'Knowledge_Of_the_Ancients',
'Green_-_Junon_Cannon_Relocated',
'Sister_Ray',
'Weapon_Surfaces_-_Value_Of_a_Life_-_Epic_Battle',
"Fire_-_Weapon_Falls_-_Paths_Opened_-_President's_Redemption",
'Imminent_Explosion_-_Hojo_Schemes_-_Midgar_Bound',
'Parachute_Into_Midgar_-_Turk_Settlement_-_The_Proud_Clod_-_Hojo_Transforms',
'Seven_Days_-_We_Have_A_Reason_To_Fight',
'Understanding',
'Last_Morning',
'We_Shall_Fight_Together',
'The_North_Crater',
"The_Planet's_Core_-_Jenova-SYNTHESIS",
'Judgment_Day_-_The_One-Winged_Angel',
'Ghost',
'The_Beginning_of_the_End',
'The_End_of_the_Beginning']


base = "http://www.finalfantasyquotes.com/ff7/script/"

pageNum = 0
for page in pages:
	print(page)
	pageNum += 1
	fileName = "raw/page_"+str(pageNum).zfill(3)+".html"
	if not path.exists(fileName):
		html = urlopen(base+page).read().decode('utf-8')
		o = open(fileName,'w')
		o.write(html)
		o.close()
		time.sleep(2)

