# Wayfinding User Test Protocol

## Objective
Validate that voice-based airport wayfinding gives accurate, useful directions to real travelers.

## Setup
- Laptop or phone running the Gradio app (localhost or shared via Gradio link)
- Stable internet for API calls
- Compiled routes in ChromaDB (run compiler before testing)

## Participants
- 3-5 people unfamiliar with the system
- Mix: frequent flyer, first-time airport visitor, non-native English speaker

## Test Script

### Pre-test (1 min)
1. "This app gives you walking directions at the airport using your voice."
2. "You'll ask it how to get from one place to another."
3. "Speak naturally, like you would ask a person."

### Tasks (5 min each)

**Task 1: Simple same-floor route**
"You are standing in front of H&M. Ask the app how to get to Check-in 2."
- Observe: Did they find the mic? Did they phrase it naturally?
- Record: Transcription accuracy, intent parsing accuracy, route correctness

**Task 2: Cross-floor route**
"You just arrived by train at the SBB station. Ask how to get to Gate A15."
- Observe: Does the system mention floor changes (elevator/escalator)?
- Record: Did the multi-step route make sense?

**Task 3: Vague query**
"You're somewhere near the shops. You need to find baggage claim."
- Observe: How does the system handle "somewhere near the shops" (no clear start)?
- Record: Did it ask for clarification or make a reasonable assumption?

**Task 4: Terminal transfer**
"You're at Gate A22 and need to get to Gate B32."
- Observe: Does it mention the Skymetro?
- Record: Total latency from speech to audio response

**Task 5: Free exploration**
"Ask it anything you want about getting around the airport."
- Observe: What do they naturally ask? What breaks?

### Post-test (2 min)
1. "On a scale of 1-5, how useful were the directions?"
2. "Was the voice clear and natural?"
3. "What was confusing or missing?"
4. "Would you use this at an airport instead of looking at a map?"

## Metrics to Track

| Metric | Target |
|--------|--------|
| Transcription accuracy | >90% words correct |
| Intent parsing success | >85% correct start+end |
| Route accuracy | 100% (pre-compiled, should be exact) |
| End-to-end latency | <8 seconds voice-to-audio |
| User satisfaction | >3.5 / 5 |
| "Would use again" | >60% yes |

## Known Limitations to Watch
- System cannot handle "I'm near the big clock" (needs exact POI names)
- No real-time location detection (GPS/beacons not integrated)
- Routes are pre-compiled; construction/closures not reflected
- English only in v1
- Some POI pairs may not have routes (unreachable or skipped in compilation)

## After Testing
1. Log all failed queries with timestamps
2. Note any POIs users expected but weren't in the system
3. Record latency for each interaction
4. Update airport_config.json if POI names need aliases
5. File issues for each failure mode
