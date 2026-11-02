# Stylised fish models: anatomy sources and design notes

These are original models made for this repo by `stylised_fish.py`. They are not copies of any game's
designs and were not modelled from game images. Real anatomy (fin layout, fin shapes, barbels, head shape,
colour zones) follows the sources below; the toy proportions, enlarged eyes, softened palette and the
retro treatments (default "ps2": jagged low polygon silhouettes with smooth shading, 128x64 bilinear
filtered body textures with painted lighting, painted fin rays, eye glints; "ps1": boxy flat shaded facets,
nearest filtered 15 bit dithered textures) are this repo's own style choices.

How the sources were read: on 2026-11-02 the anatomy sites were blocked by the session's network policy,
so the facts below come from web search result summaries of the linked pages, not from reading the pages
directly. Before quoting any of these facts on screen, open the page and confirm the wording.

All models are 1.0 unit long, facing +X with Z up. Real adult sizes differ widely; scale per scene.

## Great hammerhead, Sphyrna mokarran (`hammerhead`)

- [H1] Wikipedia, Great hammerhead, https://en.wikipedia.org/wiki/Great_hammerhead ; Florida Museum species
  profile, https://www.floridamuseum.ufl.edu/discover-fish/species-profiles/great-hammerhead/
  - Cephalofoil width 23 to 27 percent of body length; nearly straight front margin in adults with
    medial and lateral indentations. Model: half width 0.125 (25 percent).
  - First dorsal fin very tall and strongly falcate, origin over the pectoral insertions, rear tip in
    front of the pelvic origins.
  - Second dorsal and anal fins relatively large, with deeply notched rear margins.
  - Adults unmarked (juveniles may have a dark second dorsal tip).
- [H2] Colour: dark brown to light grey to olive above, fading to white below (Wikipedia, Florida
  Museum); FishBase summary via FAO guides: light grey or grey brown above, white below,
  https://fishbase.se/summary/914 . Model: grey brown back, off white belly.
- Style choice, not from a source above: heterocercal (long upper lobe) caudal fin outline.
- Painted gill slits (five) on the ps2 texture: generic shark anatomy, not from a source above; confirm
  the count for this species before showing it as a fact.

## Blue marlin, Makaira nigricans (`blue_marlin`)

- [M1] Fishes of Australia, https://fishesofaustralia.net.au/Home/species/2538 ; FAO species guide,
  https://www.fao.org/docrep/pdf/006/ad471e/ad471e56.pdf ; ICCAT manual,
  https://iccat.int/Documents/SCRS/Manual/CH2/2_1_6_BUM_ENG.pdf
  - Bill very stout and round in cross section; body elongate, not very compressed.
  - First dorsal: front lobe lower than the greatest body depth; second dorsal small.
  - Pectorals long and narrow. Sources disagree on whether they fold (Atlantic vs Indo Pacific forms).
- [M2] Colour: blue black above shading to silvery white below; about 15 bars of pale cobalt round dots
  or narrow bars on the flanks; first dorsal blackish or dark blue; other fins blackish brown.
- Style choices: bill length, crescent caudal outline, pelvic fin shape.

## Japanese sea bass, Lateolabrax japonicus (`sea_bass`)

- [S1] Wikipedia, Japanese sea bass, https://en.wikipedia.org/wiki/Japanese_sea_bass : slightly forked
  tail; large mouth with the lower jaw projecting beyond the upper; elongate and compressed body.
- [S2] Same page: first dorsal 12 to 15 spines, then a second dorsal of 12 to 14 soft rays. Young fish
  have small black spots on the back and dorsal fin that larger fish usually lose. Model: adult, no spots.
- [S3] Colour, low confidence: Sushi Walker dictionary, https://sushiwalker.com/en/dictionary/90805/ :
  dark blue green head, dark blue tinged back, silvery white from flanks to belly. No stronger colour
  source was found. ZooKeys 2019 (https://zookeys.pensoft.net/article/32624/) notes the three
  Lateolabrax species are hard to separate by shape.
- Species choice: "sea bass" is a broad common name; this model is the Japanese species.

## Amur catfish, Silurus asotus (`catfish`)

- [C1] FishBase morphology summary, https://nrm-fishbaselinux01.nrm.se/physiology/Silurus_asotus ;
  Planet Catfish, https://www.planetcatfish.com/common/silurus_asotus
  - Adults have two pairs of barbels: maxillary longer than the head; mandibular about 1/5 to 1/3 of the
    maxillary length (larvae have three pairs).
  - Long anal fin (FishBase: 67 to 84 soft rays) joined to the caudal fin.
- [C2] Small dorsal fin (FishBase: 4 soft rays). Elongate body, oval in cross section. Dark grey to
  greenish brown back, pale belly, irregular spotting on the sides.
- Species choice: "catfish" is a broad common name; this model is the Amur catfish.

## Common carp, Cyprinus carpio (`carp`)

- [K1] Wikipedia, Common carp, https://en.wikipedia.org/wiki/Common_carp ; Iowa DNR,
  https://programs.iowadnr.gov/bionet/Fish/Species/21 : downturned mouth with two pairs of barbels;
  long dorsal fin with a serrated leading spine; stout serrated spine at the front of the anal fin.
  Sources differ on exactly which anal ray is serrated.
- [K2] Missouri Department of Conservation, https://mdc.mo.gov/discover-nature/field-guide/common-carp ;
  Maryland DNR, https://dnr.maryland.gov/fisheries/Pages/fish-facts.aspx?fishname=Common+Carp :
  deep body arched toward the dorsal fin; long dorsal fin (17 to 21 rays); brassy olive back and sides,
  yellowish white belly; large dark edged scales with a dark spot at the base; forked caudal.
- [K3] Virginia DWR, https://dwr.virginia.gov/wildlife/fish/carp/ (barbels); lower fins often reddish
  (search summary of state agency guides; confirm before quoting).
