# Mid-Semester Presentation Script

*Written to be spoken aloud, in plain English. Roughly 10–11 minutes at a natural pace — trim
the "optional" asides if your slot is shorter (section 3's t-SNE walkthrough and section 7's
detection-integration detail are the easiest cuts; section 6's building result is worth keeping,
it's the strongest single finding in the deck). Suggested slide/figure cues are in [brackets].*

---

## 1. Opening — what problem are we solving? (~45 sec)

Imagine you show an AI a picture of a car. It should say "car." Now imagine you show it a car
photographed from a satellite, straight down, tiny and blurry. Then you show it the same kind of
car photographed by a drone flying close by, at an angle, in sharp detail. Does the AI still
recognize both as "car" in the same way — or does its internal understanding of "car" quietly
shift depending on how the picture was taken?

That question is the center of our project. It matters because AI systems for things like
disaster response, surveillance, and mapping increasingly need to work across satellites and
drones interchangeably — and if the AI's understanding of objects isn't consistent across
altitudes, that's a real, practical failure mode.

We call this problem **semantic drift**: the idea that the same object, in the same category,
can end up looking "different" to an AI depending on the altitude and viewpoint it was
photographed from.

[SLIDE: title + the two example images — satellite car vs. drone car]

---

## 2. How we measure it (~1 min)

To study this, we don't just eyeball images — we measure it directly. Modern AI models like
CLIP can turn any image into a list of numbers, kind of like a fingerprint, that captures what
the AI "thinks" is in the picture. Two images of similar things get similar fingerprints; two
images of different things get very different fingerprints.

So here's our method: take many satellite pictures of, say, cars, and many drone pictures of
cars. Turn each one into its numerical fingerprint. Average all the satellite fingerprints
together, and average all the drone fingerprints together. Then measure the distance between
those two averages. A bigger distance means the AI's idea of "car" has drifted more between the
two altitudes.

We ran this using two real datasets: **DOTA**, a satellite imagery dataset, and **VisDrone**, a
drone imagery dataset.

[SLIDE: diagram of the pipeline — image → fingerprint → average → compare]

---

## 3. Where we started (~1 min)

This project didn't start from scratch — earlier work had already been done on a cloud
computing platform, and it found early evidence that this drift is real: objects from the same
category landed in noticeably different regions of the AI's "fingerprint space" depending on
whether they came from the satellite dataset or the drone dataset.

Our first job was to move this whole project off the cloud and onto a local machine so we could
keep developing it independently, and then to **reproduce that original finding ourselves** —
not just trust it, but verify it. We rebuilt the pipeline locally and got results in the same
range as the original. That confirmed the phenomenon wasn't a fluke of one particular cloud
environment — it's real and repeatable.

[SLIDE: results/phase1_local/tsne_drift_local.png]

This chart is actually one of the clearest pieces of evidence in the whole project. Every dot is
one image's fingerprint. Circles are satellite images, X's are drone images. Notice that the
plot splits almost entirely into a left half and a right half — and that split lines up almost
perfectly with satellite versus drone, regardless of what object category the color represents.
In plain terms: **the AI's fingerprint space cares more about which altitude a photo came from
than what object is actually in it.** That's semantic drift, visualized directly.

---

## 4. A detective story: finding a mistake in our own earlier work (~1.5 min)

Here's where the project got more interesting. Alongside that first result, there was an earlier
attempt to answer a deeper question: if we use an AI model specially trained on satellite
imagery, does that reduce the drift? The earlier attempt compared a satellite-specialized model
against a *separately, differently* trained drone-specialized model, and forced their results
together using a mathematical shortcut to make the numbers line up.

When we examined this carefully, we realized it was **methodologically broken**. It wasn't
comparing "does specialization help" — it was really just comparing two unrelated AI models
against each other, patched together with a shortcut that doesn't actually produce a meaningful
number. We could have just kept using those numbers. Instead, we diagnosed exactly why they were
wrong, documented it clearly, and rebuilt the comparison properly.

We think this is worth presenting explicitly, not hiding: **part of doing this project well was
catching our own earlier mistake and fixing it, rather than building further on a shaky
foundation.**

[SLIDE: brief before/after — "two different models forced together" vs. "one model tested
consistently on both domains"]

---

## 5. The fixed, trustworthy result (~1.5 min)

Once we rebuilt the comparison properly — testing each AI model *consistently* on both
satellite and drone images, using real, verified labels instead of guesses — we got a clean,
defensible result.

[SLIDE: results/remoteclip_vs_clip/three_way_drift_comparison_heldout.png]

We tested three versions of the AI model side by side, on identical data:

- A general-purpose model with no specialization at all.
- **RemoteCLIP**, a model pretrained specifically on satellite and remote-sensing imagery.
- A version of the general model that **we fine-tuned ourselves**, specifically on drone
  imagery, so the comparison would be fair in both directions — not just testing satellite
  specialization, but drone specialization too.

The result: specialization consistently reduces drift. The satellite-specialized model showed
12 to 23 percent less drift than the general-purpose model, depending on the vehicle type. And
our own drone-specialized model did even better — up to 63 percent less drift for large
vehicles. In plain terms: **teaching the AI more about the specific kind of imagery it's looking
at makes its understanding of objects noticeably more consistent across altitudes.**

One more honest note, in the same spirit as the earlier detective story: our first pass at this
drone-specialized number showed an even bigger improvement, but checking it carefully turned up
two problems — a mislabeled category mixing bicycles in with cars, and a data leakage issue where
the model was being tested on images it had already seen during training. We fixed both — split
the drone dataset into a training portion and a completely separate, never-seen testing portion —
and reran everything. The result held up: still the lowest drift of the three, on genuinely
unseen data. The large-vehicle number barely moved after the fix, meaning it was already solid;
the small-vehicle number came down somewhat, which is exactly what you'd expect once
memorization is ruled out.

---

## 6. Pushing further: more object categories, a second and third dataset (~2.5 min)

One limitation we ran into: DOTA and VisDrone only share **two** object categories in common —
small vehicles and large vehicles — because satellite images can't resolve pedestrians, and
drone images don't capture things like sports fields the way satellite images do. That limits
how rich our comparison can be.

So we connected a third dataset, **xView**, which has a much more detailed set of vehicle
categories — separate labels for cars, buses, and trucks instead of one broad "vehicle" bucket.
One practical detail worth mentioning: xView is normally a 20-gigabyte download, and our machine
didn't have that much space to spare. Instead of downloading it, we found a way to **stream** it
— pulling only the images we actually need, on demand, without ever storing the full dataset
locally.

[SLIDE: results/xview_vs_visdrone/four_class_drift_comparison.png]

Comparing xView against VisDrone the same careful way as before, on car, bus, and truck, gave us
the same result again: the satellite-specialized model showed 14 to 17 percent less drift than
the general-purpose model on every one of those three categories. That's an important
confirmation on its own — it's a completely different satellite dataset from DOTA, so this shows
the finding isn't an accident of one particular dataset pairing.

Then we pushed one step further. VisDrone has no "building" category at all — it only tracks
traffic and pedestrians. So we brought in a **fourth dataset**, **UAVid**, a drone dataset that
does label buildings, and paired it with xView's building category specifically. This needed a
different extraction method too, since UAVid marks objects as painted-in colored regions on the
image rather than boxes around them, so we built a way to automatically trace out those regions.

**This produced the most striking result in the whole project.** Building drift was larger than
any vehicle category we measured, in either dataset pairing — and the satellite-specialized model
cut that drift by 30 percent, the largest reduction of anything we tested. It makes sense once
you think about it: a building looks almost nothing alike from directly overhead versus from a
low drone angle — you're comparing a roof outline to a full facade with windows — while a car or
truck keeps roughly the same shape and color no matter what altitude you look at it from. This is
arguably the single clearest piece of evidence in the whole project for why semantic drift
matters.

---

## 7. The bigger picture: connecting to real object detection (~1 min)

Everything so far measures drift using object crops that are already known and labeled. That's
good for measuring the *phenomenon*, but it's not yet a real object detector — a real system
also has to *find* objects in a full image before it can classify them.

That's where this connects to the rest of the team's work: combining all three datasets and
applying **SAM**, a model that finds object regions in an image automatically, without needing
labels. The plan is to feed SAM's detected regions into our fingerprinting and drift-measurement
pipeline — so SAM finds *where* the objects are, and our system identifies *what* they are and
measures how consistently. Together, that becomes a real, working open-vocabulary object
detector across altitudes — which is the larger goal this whole project is building toward. This
integration is in progress, not complete yet.

---

## 8. Why this matters (~30 sec)

Stepping back: open-vocabulary object detection lets AI systems recognize objects using plain
language descriptions, without needing to be retrained for every new category. That's valuable
for things like disaster response, environmental monitoring, and surveillance — situations where
you often don't know in advance every object type you'll need to detect, and where the same
system might need to work from a satellite one day and a drone the next. Our work shows that this
switch in altitude isn't free — there's a real, measurable cost to it — but also that the cost
can be reduced with the right kind of model specialization. That's a concrete, useful finding for
anyone trying to build a system that has to work across both.

---

## 9. What's next, and closing (~45 sec)

Going forward: we're checking how much our drone-specialized result moves across a few different
random samples of the held-out data, scaling our comparisons up to the full datasets, recovering
the fourth vehicle category we had to drop for now, and integrating with the SAM-based detection
pipeline so we can measure not just drift, but whether drift actually predicts real detection
failures. Longer term, we want to build a component that actively reduces this drift, rather
than just measuring it.

To summarize in one line: **we confirmed that AI models don't understand objects the same way
across satellite and drone imagery — across two independent satellite datasets and two independent
drone datasets — we found and fixed real methodology flaws along the way instead of building on
top of them, and we showed that specializing a model toward the right kind of imagery measurably
closes the gap, most dramatically for buildings.** That's where we are, and that's what we're
building on next.

Thank you — happy to take questions.

---

## Quick reference: key numbers, if asked

**DOTA vs. VisDrone (3-way):**

| Object type | General model | Satellite-specialized | Drone-specialized (ours) |
|---|---|---|---|
| Small vehicle | 0.246 | 0.189 (↓23%) | 0.159 (↓36%) |
| Large vehicle | 0.260 | 0.228 (↓12%) | 0.096 (↓63%) |

*(Lower number = less drift = more consistent understanding across altitudes. All three tested
on the same 300 images per category per dataset, on data the fine-tuned model never saw during
training, for a fair comparison. These are the corrected numbers, after fixing the category
mapping and data leakage issues described in section 5.)*

**xView vs. VisDrone/UAVid (4-way vocabulary):**

| Object type | General model | Satellite-specialized | UAV-side source |
|---|---|---|---|
| Car | 0.356 | 0.302 (↓15%) | VisDrone |
| Bus | 0.326 | 0.280 (↓14%) | VisDrone |
| Truck | 0.284 | 0.237 (↓17%) | VisDrone |
| Building | 0.542 | 0.378 (↓30%) | UAVid |

*(300 images per category per dataset. Building pairs against UAVid instead of VisDrone, since
VisDrone has no building category.)*
