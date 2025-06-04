import requests

# Ollama prompt for summary or Q&A
def run_ollama(prompt, model="mistral", temperature=0.7):
    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": model,
            "prompt": prompt,
            "temperature": temperature,
            "stream": False
        }
    )
    return response.json()["response"]

# Prepare prompt
query = """
give me pointers Widening to two lane with paved shoulder of Raipur- Jassakhera
"""

combined_context = """
--12024//3/
fic23-05-202y
3y
faqy:-
Widening to two lane with paved shoulder of Raipur-jassakhera from
km0.000to33.900
NH-458
(proposal
no.
FP/RAJ/ROAD/451083/2023)
yRi:-
319523.18.9.2024
39 fi y  
.33
..3
y133522.9.20153195692
   f    f
3.763,774875912
69.363y3
FCA Gen.

(Status as on 08.10.2024) 
 
BRIEF-NOTE 
 
1. Proposal name Widening to two lane with paved shoulder of Raipur- Jassakhera from km 0.000 to 
33.900 of NH-458. 
2 Forest Clearance Proposal No. FP/RJ/ROAD/451083/2023) 
3 Forest area to be diverted:  23.52 Ha. 
4 Existing Road Length 33.900 Kms ( 32.500 Kms in Beawar Distt & 1.500 Kms in Rajasamand District. ) 
5 Proposed Project Length 30.31 Kms 
6 Status of DPR  Submitted to MoRTH Jaipur office. 
Project brief :- 
  The above work has been considered in annual plan 2024-25 of Ministry of Road Transportation & Highway, Delhi for 
construction purpose. Accordingly detailed project report has been prepared in parts. First part is having length 22.470 Km ( 
length excluding the wild life forest portion). And second part is having length 7.84 km (Proposed in elevated road in whole 
length). So total project length will be 30.31 Km after the development of this section of National highway.

Widening to Two Lane with Paved Shoulder of NH-458 (Raipur -Jasakhera from - Km.0.0 to Km.
16.75 & Km.24.05 to Km.29.77 on EPCMode in the State of Rajasthan
List of participants is attached at Annexure-2A
2.1
Salient Features
(i)
Project Features
Sr.No.Description
Details
Length
22.470 km
Major Bridges
2Nos
3
Minor Bridges
2 Nos
Culverts
48 Nos
Flyover/Overpass
NIL
Elevated Corridor
NIL
9
7
ROB
01 No.2X76+13x20
8
RUB
NIL
6
VUP/LVUP/SVUP
6 Nos./0 Nos./0 Nos
10
CUP/PUP
NIL
11
Major Junctions
08 Nos (all are grade separated)
12
Minor junctions
15 Nos.
13
Interchanges
2 Nos
14
Length of Service Road/Slip RoadDetails
LHS
RHS
Service
3.885 km
3.885km
Road
15
Toll Plaza
At Km 8.30
16
Re-alignment Length
NIL
17
Bypass
10.26 Km
18
Bus shelter
10 Nos
19
Truck lay Byes/Rest Area
2 Nos
20
Proposed ROW
30 m on existing and 45 m on bypasses
(ii)
Status of Pre-Construction Activities
Land Acquisition
Particulars
Details
Sr.No.
Total Land Require Ha118.0771 Ha
Existing Land (Ha)
42.42 Ha
  """  # <- fill with your reranked top 3 docs






prompt = f"""You are a helpful assistant. Use the context below to answer the user's question clearly and precisely and give summary.

Context:
{combined_context}

Question: {query}

Answer:"""

# Get response from Ollama
answer = run_ollama(prompt)
print("🔍 Answer:\n", answer)
