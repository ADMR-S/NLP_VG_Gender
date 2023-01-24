import sqlite3

# TODO: Add checks and condition strings.
# TODO: Remove redundant SYSTEM IDs

def parseFile(fileName,parameters={},asJSON=False):


	def findStart(convo):
		for lineID in convo:
			lx = convo[lineID]
			if lx["title"]=="START":
				return(lineID)

	def dentry2DialogueLine(dentry):
		dTitle = dentry["title"]
		charName = "SYSTEM"
		if dTitle.count(":")>0:
			charName = dTitle[:dTitle.index(":")].strip()
		txt = dentry["dialoguetext"]
		#print(dentry)
		idx = str(dentry["conversationid"]) + "_" + str(dentry["id"])
		return({charName:txt, "_ID": idx})
	
	# Connect to database
	temp_db = sqlite3.connect(fileName)
	temp_db.row_factory = sqlite3.Row
	dentries = temp_db.execute("SELECT * FROM dentries").fetchall()
	
	# List of dialogue entries (with dialogue text)
	list_accumulator = []
	for item in dentries:
		list_accumulator.append({k: item[k] for k in item.keys()})
	
	convoDict = {}
	for line in list_accumulator:
		convoID = line["conversationid"]
		lineID = line["id"]
		if not convoID in convoDict:
			convoDict[convoID] = {}
		convoDict[convoID][lineID] = line

	# List of links between dialogue entries
	dlinks = temp_db.execute("SELECT * FROM dlinks").fetchall()

	dlinks_accumulator = []
	for item in dlinks:
		dlinks_accumulator.append({k: item[k] for k in item.keys()})
	
	# Convert to dictionary of links, look up origin, get destinations
	# convos can have destinations in other convos
	dlinkDict = {}
	for link in dlinks_accumulator:
		convoID = link["originconversationid"]
		originID = link["origindialogueid"]
		if not convoID in dlinkDict:
			dlinkDict[convoID] = {}
		if originID in dlinkDict[convoID]:
			dlinkDict[convoID][originID].append(link)
		else:
			dlinkDict[convoID][originID] = [link]
	
	# Build out, one conversation at a time
	out = []
	allConvoIDs = sorted([convoID for convoID in dlinkDict])
	#allConvoIDs = [322]
	for convoID in allConvoIDs:
		# name local convo IDs, shortcut to convoLinks
		convo = convoDict[convoID]
		convoLinks = dlinkDict[convoID]
		startID = findStart(convo)
		convoSeenIDs = []	
	
		# Recursive walk. Given an origin line ID, provide next steps
		def walkStructure(lineID):
			originConvID = convo[lineID]["conversationid"]
			idx = str(originConvID)+ "_"+ str(lineID)
			# If already seen ...
			if idx in convoSeenIDs:
				# ... just add a GOTO
				return([{"GOTO": idx}])
			else:
				convoSeenIDs.append(idx)
			dialogueLine = dentry2DialogueLine(convo[lineID])
			if not lineID in convoLinks:
				# end of the line
				return([dialogueLine])
			#print(convoLinks[lineID])
			destIDs = [(x["destinationconversationid"], x["destinationdialogueid"]) for x in convoLinks[lineID]]
			destinations = []
			for destConvID,destDialogueID in destIDs:
				if destConvID == convoID:
					destinations.append(walkStructure(destDialogueID))
				else:
					# Link to another conversation
					destinations.append([{"GOTO":str(destConvID) + "_" + str(destDialogueID)}])
	
			if len(destinations)==1:
				return([dialogueLine] + destinations[0])
			elif len(destinations)>1:
				return([dialogueLine, {"CHOICE": destinations}])
			else:
				# No destinations (already dealt with?)
				return([dialogueLine])

		convoOut = walkStructure(startID)
		out += convoOut
			

	if asJSON:
		return(json.dumps({"text":out}, indent = 4))
	return(out)