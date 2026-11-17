# Model Card: PawPal+ Grounded Care Guidance

## System summary

- **What it is:** a retrieval-augmented layer on top of PawPal+ (my Module 2 pet care planner).
- **How it works:** it retrieves cited tips from a folder of markdown pet-care notes using local TF-IDF (scikit-learn). It uses them to explain tasks, flag guideline violations, and propose missing tasks. There is no generative model and no API; the "generation" step assembles retrieved text.
- **Intended use:** helping an owner organize routine care for pets, personalized for my dog Chester (12-year-old Yorkie-Schnauzer mix) and usable for other dogs and cats.
- **Not intended for:** diagnosis, treatment, dosing, or any emergency decision.

## Limitations and biases

- **Knowledge is thin and unverified.** I drafted 14 short documents (53 chunks) from general pet-care knowledge. They have not been reviewed by a veterinarian, and they may be incomplete or oversimplified. Chester's numbers (e.g. 15-20 minute walks) are reasonable general guidelines, not a prescription for him.
- **Coverage bias.** The notes are mostly about dogs, with a smaller cat section and nothing for other animals. A hamster gets no guidance by design, and rabbits, birds, and reptiles would too. Small senior dogs are the best-served case because that is who I built it for.
- **Breed assumptions.** The Yorkie-Schnauzer notes list conditions common to the parent breeds. A mixed dog may have none of them, so this can create unneeded worry or false reassurance.
- **Retrieval matches words, not meaning.** TF-IDF only works when the task wording overlaps the notes. "Breakfast" needed a hand-written synonym list (feeding = meal, food) to find feeding advice. Unusual wording can fall back even though relevant notes exist.
- **The confidence score is not a probability.** It is a cosine similarity between the task and a note. A high score means the wording matches, not that the advice is right for this animal. One valid match scored only 0.20 against a 0.15 threshold, so the cutoff is a judgment call that I tuned by hand on a small set of cases.
- **Evaluation is small and written by me.** 12 harness cases and 25 tests were designed around the same notes I wrote, so they show the system behaves as intended, not that it generalizes to other pets, owners, or documents.
- **Life stage is a rough rule** (dogs 8+ and cats 10+ are "senior") that ignores size and breed.
- **Emergency detection is keyword-based.** It catches phrases like "seizure" or "poison" but would miss an emergency described in other words.

## Could it be misused, and how is that prevented?

- **Treating it as a vet.** The biggest risk is an owner following a tip instead of seeing a professional. Prevention: the fallback says "ask your vet" when retrieval is weak, emergency wording escalates to "contact a vet immediately", medication tasks always state that the app never recommends doses, and the knowledge base never contains dosing numbers.
- **Delaying care because the plan looked fine.** The app only reads what the owner types. It cannot see the animal, so a sick pet with routine-looking tasks gets routine-looking advice. This is a limitation I cannot fully remove, and I rely on the urgent-wording check and the "when to call the vet" notes.
- **Poisoned or wrong documents.** Anyone who edits `knowledge/` changes what the system says, and the system will cite those edits confidently. Mitigations: every claim shows its source file so a bad note is traceable, and suggestions need explicit approval. A fuller version would review documents before indexing.
- **Privacy.** Everything runs locally and nothing leaves the machine. Chester's personal notes are filtered so they are never shown for another pet.
- **Wrong-species advice.** Prevented by species filters; a test and a harness case check it.

## What surprised me while testing reliability

- **Passing tests on the first run did not mean the system was right.** The first demo run showed a hamster being told to brush its teeth, Chester's breakfast getting his general profile instead of his feeding notes, and a duplicate suggestion for a cat that already had a play task. Reading the real output found problems my tests had not been written to look for, so I added tests for each one afterward.
- **My own test expectation was wrong once.** I expected the hamster's medication task to fall back, but the general medication-safety document legitimately applies to any species. The system was right and my test was wrong, so I fixed the test and kept the reminder check.
- **A small amount of structure did most of the work.** Species, life-stage, and pet-name filters mattered more for correctness than ranking tweaks did.
- **Breaking the system on purpose was informative.** Disabling the filter in a scratch copy made five tests fail, which showed me those tests were doing real work.

## Collaboration with AI

I used Claude to brainstorm the design, draft the code and knowledge base, and run the tests, then I made the key choices.

**One suggestion I accepted:** I accepted a fully local TF-IDF retriever with templated, cited output. I had said I did not want to use an API yet, and this approach fit that constraint. It kept the project reproducible from a fresh clone, free, and easy to explain and test, with the planner depending only on a small retriever interface in case I later switch to embeddings.

**Suggestions I did not accept as written or that were corrected:**

- **Embeddings in the first diagram.** The first architecture sketch said "chunk + embed". Because I chose no API and no extra model dependency, we changed it to "chunk + TF-IDF" before committing it, and the final diagram matches the implementation.
- **A fuzzy rule for avoiding duplicate suggestions.** The first version skipped a suggestion when more than half of its words matched an existing task. It failed both ways: it missed that "Play session" duplicated "Wand toy play", and it risked hiding "Brush teeth" because of an existing "Brush coat" task. I replaced it with an explicit coverage key written in each document (e.g. `teeth`, `play`), which is easier to read and to test.
- **A wrong test expectation (the hamster's pill task).** Described above; I corrected the test and kept the safer behavior.
- **A general "any species" tag on the dental note.** This was too broad and let hamsters receive tooth-brushing advice. Docs can now list the species they apply to.

**How I verified correctness:** 25 automated tests, a 12-case evaluation harness, a deliberate break of the species filter to confirm the tests can fail, a side-by-side baseline-versus-RAG demo to inspect real output, and logs of every retrieval. I did not verify the veterinary content itself; that still needs a review by a vet or a reputable source.

## Evaluation log (text, reproducible with `python eval_harness.py`)

| Test input | Evaluation criteria | Result |
|---|---|---|
| Chester, 40-min walk | Grounded, cites Chester's walks note, warns about the 20-min guideline | Pass |
| Chester, 15-min walk | Grounded, cites the walks note, no warning | Pass |
| Chester, breakfast | Cites his feeding notes | Pass |
| Chester, morning pill | Grounded, medication reminder present | Pass |
| Chester, brush teeth | Cites dental guidance | Pass |
| Rex (another senior dog), 30-min walk | Uses general senior-walk note, does not cite Chester's profile, warns | Pass |
| Pip (puppy), meal | Cites puppy feeding note | Pass |
| Luna (cat), dinner | Cites cat note, no dog sources | Pass |
| Nibbles (hamster), refresh bedding | Honest fallback, no citations | Pass |
| Nibbles (hamster), pill | Grounded in general medication note, reminder kept | Pass |
| "Chester had a seizure" | Escalated as urgent | Pass |
| Gibberish description | Does not present a weak match as confident | Pass |

**Result:** 12 of 12 cases and 25 of 25 tests pass; average confidence on grounded answers is 0.33. These are tests I wrote against my own documents, so passing them is evidence of intended behavior, not of general accuracy.
