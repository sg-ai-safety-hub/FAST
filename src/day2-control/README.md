# Day 2: Control

Why alignment is slow and unreliable, and how the AI control agenda borrows from infosec to plan
for models that might not be aligned. The day gets at what a control protocol actually measures,
and where today's experiments hit their limits.

## Run of show

- [`harm-classifier/`](harm-classifier/) · **Train a harm classifier, then break it** · Lab · 60–90 min
- [`control-hackathon/`](control-hackathon/) · **Insecure-code control hackathon** · Hackathon

The lab comes first. Participants train the cheapest monitor there is, a DistilBERT classifier over
Aegis 2.0 prompts, and score it the way control scores a monitor: catch rate at a fixed
false-flag budget. Then they hunt its holes (perturbations, padding past truncation, over-flagging,
arguable labels) and patch one to see what the patch costs. It sets up the trade-off the hackathon
scores.

One hackathon, one live board, played against real models. A stream of small Python tasks arrives;
red writes a prompt that turns an untrusted model into a model organism writing insecure code, and
blue writes a monitor (a prompt plus a model it picks) that tries to catch it. Blue competes to
catch with the *smallest* model on the ladder; red competes to fool the most monitors; a strong
judge model settles ground truth. Teams swap sides partway through, so everyone attacks the
monitoring assumptions they just relied on.

Order here is the source of truth. Directories are unnumbered so reordering the run of show
doesn't rename paths or break bookmarked Colab links. Guest lectures have no directory; speaker
briefs stay in the internal docs, and any reading a session assumes goes in the exercise above it.
