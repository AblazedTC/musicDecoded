import sys
sys.path.insert(0, '.')

from services.madmom_detector_service import MadmomDetectorService

s = MadmomDetectorService()
r = s.detect_beats('data/sunnydays(g).mp3')

print(f"Success: {r['success']}")
print(f"Has candidates: {'downbeat_candidates' in r}")
print(f"Time sig: {r.get('time_signature')}")
print(f"BPM: {r.get('bpm'):.2f}")
print(f"Beats: {r.get('total_beats')}")
print(f"Downbeats: {r.get('total_downbeats')}")

if 'downbeat_candidates' in r:
    print(f"Candidates 3/4: {len(r['downbeat_candidates']['3'])}")
    print(f"Candidates 4/4: {len(r['downbeat_candidates']['4'])}")
