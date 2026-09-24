# Day 2: Control

Why alignment is slow and unreliable, and how the AI control agenda borrows from infosec to plan
for models that might not be aligned. The day gets at what a control protocol actually measures,
and where today's experiments hit their limits.

## Run of show

- [`control-hackathon/`](control-hackathon/) · **Insecure-code control hackathon** · Hackathon

One hackathon, one live board, played against real models. A stream of small Python tasks arrives;
red writes a prompt that turns an untrusted model into a model organism writing insecure code, and
blue writes a monitor (a prompt plus a model it picks) that tries to catch it. Blue competes to
catch with the *smallest* model on the ladder; red competes to fool the most monitors; a strong
judge model settles ground truth. Teams swap sides partway through, so everyone attacks the
monitoring assumptions they just relied on.

Order here is the source of truth. Directories are unnumbered so reordering the run of show
doesn't rename paths or break bookmarked Colab links. Guest lectures have no directory; speaker
briefs stay in the internal docs, and any reading a session assumes goes in the exercise above it.
