# MASTER blueprint v3: City Folk (CF, Wii 2008) and New Leaf (NL, 3DS 2012) look for Blender 5.2 EEVEE

Synthesis of A, B, C, D (v1) and their independent reviews, plus synthesis spot checks (scripts in `nook/synth/`).
v3: third independent audit pass [Au] (scripts in `nook/audit/`); every audited number and its verdict is in section 14.
Rule applied: a reviewed number replaces the v1 number unless stated; rejected claims are dropped (listed where it
matters). Provenance tags: [A] analyst, [vA] verifier of A, [S] this synthesis. Confidence: H high, M medium, L low.
Source ranking used when sources disagree: in game capture (retail) over press capture (E3 builds) over model
render over publisher artwork. "P" = player height (sole to hair top), the unit of all world distances.

## 0. Evidence base

| category | CF | NL | proves | cannot prove |
|---|---|---|---|---|
| in game capture, retail | 7 (CF#870 to 876; 870 title) | 9 (NL#937 to 945; 937 title, 941 to 945 emote crops 101x150, 939 is a 200x140 Nintendo Direct 2012 grab) | in game rendering: camera, shadows, rim, colours | NL: only NL#938 (night), NL#939, NL#940 are full outdoor frames; no open far ground, so retail NL haze is unproven |
| press capture (game renderer, pre release build) | 12 (E3 2008, CF#877 to 888) | 22 (E3 2010 NL#946 to 953, TGS 2010 NL#954, E3 2011 NL#955 to 967) | rendering of that build | final retail palette (NL E3 2010 sky is cyan, retail is deep blue: section 4) |
| model render (viewer or rip, lighting not the game's) | 129 (players 16, specials 58 incl. 2 sprites CF#395, 420, villagers 26, objects 18, Prima 11) | 333 (NL#604 to 936) | mesh sharing, silhouettes, proportions, facets, textures | in game lighting (NL viewer renders lack the in game rim) |
| publisher artwork (CG renders of the game models) | CF#22 to 309 (288) | NL#19 to 418 (400) | proportions, materials, publisher lighting | in game shading (CF art shadows far darker than in game) |
| raw texture | 346 shirts 32x128 | none | texture format, palette, UV layout | environment textures (none in corpus) |
| sprites and icons | 36 icons, 21 large icons, 2 misc (+8 icon renders) | 98 item icons, 13 question icons, 9 signs, 7 misc | UI pixel art formats | 3D textures |
| key art, print, logos, concept | scenes 7, environment 7, wallpapers 11, box 17, logos 5, misc 3, booklet 12 pages | misc art 37, concept 10, wildlife 11, box 16, logos 3, consoles 5, booklet 6 pages | thumbnail and title card language | anything about the game camera or world shape |

Totals: CF 891, NL 973 images. Native resolutions: CF frames are 640 px wide resampled to 832x456 (rows native)
[A, vA: spectral peak 4.26 to 4.34 px in 14 of 19 frames; H]; NL top screen 400x240, NL#953 and NL#937 are raw
RGB565 / 16 bit captures [A, vA; H]; NL#966, 967 are smooth 1.98x resamples [vA].
Known gaps: (1) almost all NL outdoor numbers come from E3 2010/2011 press builds; (2) NL lighting/haze in retail
rests on NL#938 to 945 only; (3) no CF in game rim measurement; (4) no environment textures, no triangle counts,
no world grid size outdoors; (5) the bend radius rests on an assumption about where the bend starts.

## 1. Camera and world

### 1.1 Camera (solved from square floor grids; focal rests on the square tile assumption [vA])

| quantity | CF | NL | source and confidence |
|---|---|---|---|
| pitch below horizontal | 35 deg (34.8 to 35.8; CF#871 outdoors 37.6) | 35 deg (34.8 to 35.1) | CF#876 in game checker [A 35.0, vA 34.8], CF#879 press [vA 35.1], CF#871 caustic tiling [A 37.8, vA 37.6]; NL#950 E3 2010 parquet [A 33.7 revised by vA to 35.0]. H (CF), M (NL) |
| vertical FOV | 22.5 deg (22.1 to 23.3) | 28 deg (25 to 31) | vA homographies; A's NL 24.7 REVISED. 10% tile non squareness moves pitch about 4 deg and FOV about 15% [vA]. [Au] third method (tile width versus row is a line that vanishes at the horizon row, plus equal tile depth spacing; `audit/cam_widthline.py`, `audit/nl950_cam.py`): CF#876 pitch 35.1, vFOV 22.8, hFOV 40.3; NL#950 pitch 35.2, vFOV 26.6 (plus or minus 5% tile widths: pitch 33.3 to 37.4, vFOV 24.8 to 28.7). NL 28 kept, it sits inside every estimate's range |
| horizontal FOV (native aspect) | 40 deg | 45 deg | vA |
| focal, 24 mm sensor height, sensor fit VERTICAL | 60.3 mm | 48 mm | 12/tan(vFOV/2) [vA]; CF 60.5 CORRECTED to 60.3 (12/tan 11.25 deg = 60.33) [Au] |
| camera height / horizontal distance to feet, interior | 5.2 P / 6.7 P (eye to player 8.2 P) | 3.0 P / 3.45 P (eye to player 4.3 P; range 3.9 to 4.8) | vA (A's 5.8/7.5 and 3.3/4.0 REVISED). CF CORRECTED from 5.4/6.9/8.5 [Au]: the CF#876 player is 25.2% of the frame (hair top row 175, sole row 290), not 24%; solving share 0.252 and feet row 0.636 at pitch 35, vFOV 22.5 gives 5.22/6.66; the width line gives camera height 6.0 tiles, x 0.87 P per tile (vA tile size) = 5.2 P. M |
| outdoor distance factor | about 1.2x interior: height 6.3 P, distance 8.4 P | about 1.2x: height 3.6 P, distance 4.1 P (3.5 / 4.15 also fits the feet row 0.68 [Au]) | player smaller outdoors: CF#871, 878 20.0 to 20.5% vs CF#876 25% [Au]; NL#946 31% vs NL#950 38% [vA]. CF CORRECTED from 5.9/7.6 [Au]: share 0.203 plus feet row 0.59 solved at pitch 35, vFOV 22.5 (`audit/arith.py`). M |
| yaw / roll | none (vanishing point on the centre column, lateral edges level) | none, except NL#959 interior at strong yaw (press) | vA H |
| lens effects | none: no DOF, no motion blur, no MSAA | same | A, vA H |

The NL lens is wider AND the camera nearer: NL closeness comes from both (eye distance ratio 2.0, player share
ratio 1.58) [vA].

### 1.2 Framing (outdoor gameplay view)

| rule | CF | NL | source |
|---|---|---|---|
| player height / frame height | 20 to 21% outdoors (CF#871 20.3%, CF#878 three players on the near row 20.0 to 20.5% [Au]; CF#873 20.8% crouching), 25% interior (CF#876 25.2% [Au]; vA 24%). The v2 "21 to 22%" was not supported by its own cited values: CORRECTED | 31 to 34% (NL#946 31%, NL#961 34%, retail NL#940 34%), 38% interior; close press views 45 to 52% (NL#947 about 50%) | A, vA, vB; H |
| player horizontal position | centred within 3% | centred | A H |
| feet row from top | 54 to 64% | 67 to 68% | A H |
| ground ends (horizon) row from top | 19.4 to 20.7% (CF#870 19.6, 873 19.4, 887 19.8, 878 20.7); [Au] automatic last sky row per column: CF#870 19.7, 873 19.5, 878 21.1 | 16.5% (NL#953 39.5 of 240; NL#946 same); [Au] 16.7 to 17.1% (NL#953 40, NL#949 40, NL#946 41 of 240) | vA H; NL#947 "25%" REJECTED (true 38%, a lower camera) |
| sky visible at centre column | 0 to 7% in gameplay (CF#871 has no sky; D's 0.17 REJECTED) | 13 to 18% at the centre, up to about 30% at the sides between trees | vD H |
| look up view | horizon at 73% (CF#882, 883) | not seen | A |

### 1.3 World curvature ("rolling log")

* Shape: cylinder with its axis left to right (bend in depth only). Horizon sagitta 0 to 3 px over 230 to 720 px
  spans in 11 frames of both games [A, vA]; a sphere would bow it. H.
* Evidence of bend: with the solved cameras a flat plane puts the horizon far above the frame (y_v about minus 580
  to minus 650 CF, minus 217 NL) yet ground ends on the rows above; far objects rise above that edge with trunks
  hidden; the CF#871 caustic depth step drifts 6% over 55 rows [vA]. The NL#961 lamp lean argument is REJECTED [vA].
* Radius, tangent model R = 2H/tan^2(alpha), bend starting under the camera: CF R about 34 to 43 P, NL about 31 P
  (vA, outdoor heights). If the bend starts at the player the radius drops to about 9 P (CF) and 12 P (NL) [A].
  [S] recomputed the player start case with the reviewed cameras, R = 2(H minus z0 tan alpha)/tan^2 alpha:
  NL outdoors H 3.6, z0 4.1, alpha 25.6 deg: R = 14.3 P (interior camera 11.7 P); CF outdoors H 5.9, z0 7.6,
  alpha 27.9 to 30.7 deg: R = 8 to 13 P (10.4 P at 29.3 deg). A Blender bend that starts at the focus point
  (as the user's beach script does) should therefore use about 12 to 14 P (NL) or about 13 P (CF).
  [Au] CORRECTED the CF value: with the pitch used everywhere else (35.0 deg) the CF horizon row 19.7% gives alpha
  28.1 deg and R = 12.9 P with either camera (v2 5.9 / 7.6 or v3 6.3 / 8.4); 10.4 P needs alpha 29.3 deg, i.e. a pitch of
  36.2 deg, which the constants do not use. NL check: alpha 25.5 deg, R = 14.4 P (v2 camera) or 13.0 to 13.4 P
  (camera refit to share and feet row); bend under the camera 30 to 32 P (NL), 41 to 44 P (CF) [`audit/arith.py`].
  So R is LOW confidence; CF and NL bends are similar. The reliable target is the horizon row: tune R until the
  ground's tangent line sits at 19.7% (CF) or 16.5% (NL) from the top of a landscape frame with the game lens.
* Official art bowls the world on both sides (NL#430 to 435 fisheye "little planet"); that is a key art
  convention, not the game projection [A, D, vD]. The NL concept map NL#467 is flat with stacked tiers [D].

### 1.4 Portrait (1080x1920) adaptation, a design choice, not a measurement
Two consistent options [vA]: (a) same subject scale as the game's frame height: keep vFOV 22.5 (CF) or 28 (NL)
in portrait: very tight, hFOV 13 to 16 deg; (b) same width as the game: hFOV 40 (CF) or 45 (NL) deg across the
1080 px width (sensor fit HORIZONTAL, focal 36 mm sensor width: 49.5 mm CF, 43.5 mm NL), vertical FOV about 66
to 73 deg (65.8 CF, 72.7 NL [Au arithmetic]), which shows much more ground and sky. Recommendation [S]: option (b) for gameplay explainers, retuning the bend radius so the ground ends 17 to 20%
from the top. Consequence (arithmetic): player height / frame WIDTH is 0.205 x 456/832 = 0.11 (CF) and
0.32 x 240/400 = 0.19 (NL), i.e. about 121 px (CF) or 207 px (NL) tall in a 1080 px wide frame (6% and 11% of
1920) [Au: CF updated for the corrected 20.5% share]. For readable shorts, a closer "NL press close view" (player 45 to 52% of the game frame height, NL#947,
951, 956, 964) is evidence backed and gives about 290 px; use it for technique close ups.

## 2. Lighting and shading

| item | spec | source and confidence |
|---|---|---|
| key direction | from the camera side and above ("headlight" key): brightest band central on round forms, falloff toward every silhouette edge; top facing rock faces brightest. No left or right bias in NL (left brighter in 52% of 333 renders) and only slight in CF renders (79% left brighter, ratio 1.018) | A profiles CF#882, 878, NL#943, rocks CF#873/NL#951; vB left/right test. Azimuth "within 15 deg" UNVERIFIABLE [vA]. B's "front left key" REJECTED for NL [vB]. M |
| smooth vs steps | smooth Gouraud like gradients, no toon bands, no 2 or 3 step ramp anywhere; low poly facets visible as flat planes on some props (NL#951 rock) | A, B, D, vD (p50/p95 0.68 to 0.83 in every set). H |
| shadow side / lit, in game (V p5/p95 inside one material) | about 0.50 to 0.55 in BOTH games: NL hair median 0.56 (n 5), NL boulder 0.52 to 0.54; CF hair median 0.49 (n 4), CF boulder 0.59 | vD conflict 2 (in game, NL#941 to 951, CF#873, 878). Painted darks pull p5 down, so these are lower bounds. [Au] own brown hair mask: NL#941 0.41, 942 0.53, 944 0.56, 945 0.44 (NL#943, back of head, 0.70), median of five 0.53; CF#878 two boys 0.44, one 0.74: CONFIRMED. H |
| shadow side / lit, publisher art only | CF players and specials 0.24 to 0.31 (deep), CF villager art 0.44 to 0.45, NL art 0.41 to 0.46; model viewer renders NL 0.49, CF 0.31 to 0.50 | vD. "CF much darker than NL" is an ARTWORK difference only; use 0.25 to 0.30 only to imitate CF promo renders |
| shadow hue (rock, same model in both games) | warm lit #CAAFA7 (CF#873) / #D6C3BC (NL) to violet shadow #5E5A78 (CF) / #535369 (NL); hue swings from about 14 to about 245 deg | vD, A (#C6B2AB / #5A5875), vA; [Au] brightest 10% / darkest 10% means: CF#873 #CAB0A9 / #5F5C7A, NL#946 #D7C5BE / #605F78, NL#951 #D8C5BD / #4C4E60: CONFIRMED. H: implement as warm key plus violet ambient, never grey darkening |
| per channel shading LUT on that rock, relative to the material median [S] | shadow (dark 10%) NL (0.535 to 0.550, 0.554 to 0.574, 0.647 to 0.660), CF (0.587, 0.645, 0.790); light (bright 10%) NL (1.46, 1.38, 1.20), CF (1.34, 1.31, 1.13) | `synth/lut_rock.py` on CF#873, NL#946, NL#949 (press, same rock model). Texture variation included, so the light end is an upper bound. [Au] own mask with 1 px erosion (`audit/lut_rock3.py`), light end CONFIRMED within 0.04: NL#946 (1.44, 1.36, 1.18), NL#949 (1.46, 1.35, 1.19), new sample NL#951 (1.45, 1.35, 1.17), CF#873 (1.30, 1.29, 1.12). Shadow end depends on the mask: eroded NL#946/949 (0.64 to 0.65, 0.65 to 0.66, 0.74 to 0.75), NL#951 (0.51, 0.54, 0.59), CF#873 (0.62, 0.67, 0.81); v2 values sit inside that spread. M |
| character hue shift in shadow | warm fills 1 to 10 deg toward red (set dependent), saturation +0.05 to +0.15; cool fills +2 to +3 deg toward violet; player skin toward yellow in CF (+5 to +8) | vD, vB. NL whites shade lavender (#A69AB8 on NL#880 render), CF whites neutral grey (#827E80 CF#382): render only, L as an in game trait |
| foliage hue shift | darks toward teal: NL#419 p5 #0C3D26, p50 #0E842E, p95 #4CE279; CF key art canopy dark #0C5041 (hue 167) to light #81B424 (hue 81) | D, vD. H for direction |
| rim light, NL | YES in game: 1 to 2 px (1 to 2% of figure height), plus 15 to 35% luminance, tinted pink lilac. NL#941 1.15, 942 1.19, 945 1.19, 943 1.33; NL art 1.13 to 1.22 | vD conflict 1 (retail emote captures). NL model viewer renders lack it (1.03): do not use them as NL lighting reference. [Au] profile down from the top of the hair (`audit/nl_rim.py`): the second pixel row reads 1.32 (NL#941), 1.56 (942), 1.50 (944), 1.32 to 1.50 (945) of the hair 4 to 8 px inside, the third row 1.05 to 1.3; blue rises there (hair interior B 76 to 81, row 1 B 94 to 122) while the lime grass has B near 20, so this is a lit rim and not only grass bleed. Head top rims are therefore stronger than the ring average: gain range CORRECTED to plus 15 to 50% (ring average 15 to 35%, head top up to 50%). M to H |
| rim light, CF | cannot confirm in game; CF artwork has the opposite (silhouette 0.74 to 0.82 darker, broad falloff, not a line) | A, vD. Use none |
| specular | none on matte characters or terrain; small painted or soft highlight on glossy props (NL fruit soft blob, CF fruit hard white disc, by eye) | A, B, D. M |
| distance haze, NL press builds | ground brightens and yellows over the far 40 to 50% of visible ground: NL#953 L 99 to 104 near to 182 at the horizon; far colour typical #94CC78 (max #AAE677 NL#956); independent of sky colour; water too (NL#957 near #3B6CAC, far #96B8F9). Horizon nearly dissolves (sky L 191 vs far grass 177 to 197) | A, vA (4 press images). Retail NL haze UNPROVEN here (no open far ground in NL#938 to 940). SUPERSEDED: confirmed on 22 retail frames (16.3) as a ground and water layer only, objects keep their colour (16.17). M |
| haze, CF | none: grass L constant within 4 to 6 levels from horizon to bottom (CF#873, 870, 882, 888); crisp dark sky against bright grass step (L 89 to 129) | A, vA. H |
| night | CF night grass to about 0.6 of day luminance and hue 155 to 173 deg; characters multiplied about (0.35, 0.37, 0.82); NL night blue violet with warm point lights and glow (lamps, windows) | A (CF#883, NL#938, NL#961); character ratio not re measured. M |
| exposure | scenes almost never clip: near white median at most 1%, near black at most 0.2% | C, vC. H |

## 3. Shadows (characters and objects)

No sun directed cast shadows from characters in any capture; all character shadows are blob decals centred under
the feet with no offset (A, vA, H). CF#882 balloon shadow straight down (by eye).

| spec | NL (retail in game NL#941, 942, 944, 945; 941 and 944 are the same frame) | CF (press CF#877, CF#880; in game CF#876 by eye) |
|---|---|---|
| core multiply (luminance) | 0.41 to 0.50, use 0.48 (A's 0.55 to 0.61 REVISED) | 0.28 to 0.60 by image and row (CF#877 0.35 to 0.37, CF#880 0.60). "CF darker than NL" REJECTED [vA] |
| tint (per channel multiply) | R and G equal, B lifted: grass (126,201,8) to (85,138,13) | violet: CF#877 (0.355, 0.356, 0.565), CF#880 (0.60, 0.59, 0.73); blue multiplied far less |
| width at 50% / full extent, in P | 0.38 P / 0.43 P (A 0.30/0.38 REVISED) | 50% width about 0.40 P (40 to 46 px vs about 110 px players); 10% extent 0.6 to 0.7 P |
| profile | quick drop to about 0.7 over 3 px, then a gentle cone to the core; 10 to 90% edge about 10 px (about 0.09 P) | soft plateau plus long ramps; 10 to 90% edge about 22 px (0.2 P); not a 6 px cone |
| on screen aspect | 1.3 to 1.4 : 1 (A's 2:1 REVISED) | similar (by eye) |
| trees and props | EVERY grounded object gets one (section 15): canopy shadow about 0.65 of the canopy width, centred under the canopy (no sun offset), soft irregular edge, luminance 0.77 to 0.84 of open grass near the rim and darker toward the trunk (NL#946); ground under flower clumps 0.68 (NL#946); rock contact band dark, extending past the rock (NL#946, 951). Same dark blue grey blend as the player blob (R and G fall more than B). M (one E3 2010 frame for the numbers, by eye on NL#951, 953) | cannot confirm |

Confidence: NL H (four retail frames); CF M (press plus one in game frame by eye).

Audit [Au] (`audit/nl_blob2.py`: horizontal profile at the sole row, normalised by grass more than 40 px to the side):
* NL core luminance at the sole row 0.43 (NL#941), 0.45 (942), 0.41 (944), 0.49 (945), 0.51 (943): CONFIRMED 0.41 to 0.50.
  Width at 50% 42 to 46 px against a figure of about 109 px (NL#941 hair top row 22, sole 130): 0.39 to 0.42 P: CONFIRMED
  (0.38 P).
* NL tint: R and G ratios equal within 0.04 (0.42 to 0.53); blue rises in ABSOLUTE terms (B ratio 1.2 to 2.8, because lime
  grass B is about 20 of 255). A pure multiply cannot raise a channel, so the v2 constant (0.48, 0.48, 0.62) cannot
  reproduce it on lime grass: CORRECTED the recipe to an alpha blend toward a dark blue grey (blend colour and alpha are
  not separable from these frames: estimate, tune on a test frame until R and G fall to 0.45 of the grass and B rises).
* CF#877 below the feet, darkest 10%: (0.36, 0.40, 0.73), luminance 0.43 (v2: 0.355, 0.356, 0.565): violet tint CONFIRMED,
  values sampling sensitive; keep the CF band 0.3 to 0.6.
* Colour space (arithmetic, CORRECTION to the recipe): every ratio above is a ratio of sRGB pixel values. Blender
  multiplies in linear light, so a displayed 0.45 to 0.48 on these colours needs a linear multiply of about 0.17 to 0.22
  (lime grass G 0.173, R 0.184, mid grey 0.219 with the exact sRGB curve; 0.48^2.2 = 0.20). Likewise a displayed rim of
  plus 22% on hair (sRGB 88) is a linear gain of about 1.52.

## 4. Terrain, water, sky, seasons, time of day

### 4.1 Sky (build matters for NL)

| scene | top / body | horizon band | source, confidence |
|---|---|---|---|
| CF day (E3 2008 press) | body #1A2AD0 (per frame medians #1424CF to #1C36CE, hue 231 to 236, S about 0.88, V about 0.81); zenith down to #040CBF in the darkest frame CF#882 | lightens only in a thin band about 60 px (13% of 456) above the horizon: #0C30D7 to #0D52E3. A full height gradient is wrong | vC (8 frames), vA (CF#882 #0815CA until 60 px above horizon). C's #040FC3 REVISED: one frame was 90% of the pooled pixels. [Au] rows 0 to 30: CF#870 #1A22D2, 871 #1D31D1, 873 #1D30D1, 878 #1A23D2; CF#882 #030CBE: CONFIRMED. H |
| CF rain | overcast #748AA0 (S 0.28) | | vC (CF#887). M |
| CF night | #000C3B | #002C6C | vA (CF#883). H for that frame |
| CF city | pale horizon #7B94BC | | vC (CF#877). M |
| NL E3 2010 press | bright cyan #45C5FF (top #3BBBFF) | #50D0FF to #58D4F8 | vC, A, vA (NL#946, 947, 949, 951, 953); [Au] rows 0 to 40 (body plus band) #48CCF8 to #4DCDFF: CONFIRMED. H for that build |
| NL E3 2011 press | saturated blue to violet #213EED (NL#963), #3356F1 (957), #6378EB (956), #6D4FFA (955), #7942FE (964, possibly evening) | | vC. H for that build |
| NL retail / near retail | deep blue #2135E7 (h 234, S 0.86, V 0.91) on retail NL#940; #242CE6 on NL#939 (Nintendo Direct 2012); clouds (NL#939): thin soft low contrast wisps, core #5790D9, brightest #6FA2E1, against #3044DF, not white puffs (white puffy clouds are key art only). L (one 200x140 frame) | lower sky #4078DA to #6894D9 near a cloud (NL#939) | [S] `synth/sky_retail.py`, vC; [Au] NL#940 #2139E8, NL#939 #2323EF (strict blue mask): CONFIRMED within 9 levels. M (two small frames). Use the deep blue for an NL retail look; cyan only for an "early NL" look |
| NL snow night, evening | snow #4E56B1 under night light (NL#961); evening sky #7942FE (NL#964) | | vA. M |

### 4.2 Ground and grass

| role | CF | NL | source |
|---|---|---|---|
| grass | #009758 (CF#882), #009158 (CF#888), #0A8564 (CF#873): hue 154 to 166, V 0.47 to 0.57, teal green | press builds #188050 / #147D4D (hue 145 to 154); RETAIL emote captures lime: median #92DB10 (hue 81 to 82, S 0.93, V 0.86), mottle from p25 #82CC09 to p75 #A1E917 (NL#941, 942, 944, 945, all grass pixels incl. the blob shadow) [Au]; v2 #82CD0C is the dark quartile, CORRECTED | A, vA, vC, vD, Au. Season or build behind the lime cannot be confirmed. CF [Au]: CF#882 #009658, 888 #009158, 873 #068464, 878 #009B59: CONFIRMED. NL press [Au] #127C4B to #1D8551 (NL#946, 953, 951): CONFIRMED. M |
| far grass | same as near (no haze) | haze to #94CC78 (press); retail toward #88BF68, far over near 2.15 (16.3) | A, vA |
| dirt / path | #BC904A to #CDA349 (A), #C69342 (C); hard edged blobs with leafy, star cut edges | #906826 path (C, NL#940); plaza ground golden brown cobble (NL#940 by eye) | A, C. M |
| sand | #E7CB8B (CF#871, S 0.40) | #FED08A (NL#949), #CDA177 (NL#963) | vC; [Au] CF#871 #E7CB8B exact, NL#949 #FED38C: CONFIRMED (NL#963 #D7A87E with a lighter mask). H for frames |
| material boundaries | HARD: dirt to grass in 1 px (CF#888) with a dark teal outline (0,41,62) 2 to 3 px wide around grass shapes (A's 4 to 5 px REVISED) | soft, blending over a few px (by eye) | A, vA |
| ground motif | bold decal shapes (leaf, star, triangle, round patches; varies per frame) | fine small triangle print on a mottled base (in game NL#949); key art changes the motif per season: triangles (summer), circles/dots (spring), warm orange/magenta/yellow triangles (fall), stars on snow (winter) | A, D, vD (by eye). H existence, M per season mapping |
| autumn ground | not sampled | about #754E4F (NL#962; A's #8C6457 REVISED) | vA |
| cliffs and banks | vertical walls, pinkish beige #CEA88A with evenly spaced dark violet vertical stripes, sand rim then grass | tall pale salmon walls #FFCEAE lit, #BA8B8B shaded, horizontal strata; grass edge rounds over the drop with a lit rim | A (not re measured, consistent by eye [vA]). M |
| rock | mauve grey #A28E9A (CF#873) | #9992A2 (NL#951, 946) | C (not re measured). M |

### 4.3 Water

| item | CF | NL |
|---|---|---|
| colours | sea #25549A (CF#871), near bank #3275A7, river #2886C5 / #2B6DA5, shallow #2A88AF | sea #3A75B4 to #3D7ABB; shallow shore #6BBEF0; far sea #96B8F9 (haze) |
| caustics | tiled cellular line texture; period 185 px (y 250) to 227 px (y 450) in an 832 px frame; lines +18% at p95, +39% at p99 over the water median; soft 15 to 20 px ramps | cellular, period 115 to 120 px at y 210 to 230 of 240 (NL#960); gentle swells |
| opacity | opaque, no visible bottom | TRANSLUCENT near the surface: submerged posts and tyres stay visible (NL#957, 967, press). NL#957 column through a post: wood #F5B272 above the line, then #A0A5AB, #808BA4, #5F7BA5 just below, reaching the water #3B6DAC within about 12 px (roughly 0.2 P): a pale blue grey cast first, then the water colour by depth. Fish read as dark shapes under the surface. M (one press frame) |
| shoreline (CF#871 x 720, top to bottom) | dry sand (231,203,139); light wet (231,215,163) 16 px; darker wet (211,195,151) 12 px; foam (168 to 190, 221 to 224, 218 to 230) 3 to 4 px; smooth ramp to (39,103,152) over 40 px | white foam band at least 5 px at the sand line (NL#949) |
| source | A, vA (H for CF#871; NL M, press); [Au] CF#871 sea #25539A: CONFIRMED | A, vA, vC; [Au] near water rows 120 to 240: NL#957 #396FAD, NL#960 #3A72AF, 3 to 5% darker than #3A75B4: CONFIRMED within 8 levels |

### 4.4 Seasons and time of day (key art colour scripts, publisher art, not in game; D, mostly not re measured)
CF scenes are ONE layout recoloured six times (CF#329 to 334); NL key art is one layout repopulated per season
(NL#432 to 435). Sky top to horizon: CF spring #39BDEE to #96CEEE, summer #35B1E0 to #7DC0BB, fall lilac #A78CBB
to #CB8CB3, winter #B9CFE5 to #C5E1EE, fireworks night #333C6E to #3E487A; NL spring #83C7FC to #57DBFF, summer
#4A86FF to #A6EAFF, fall dusk #7616A5 to #AF428F to #BE585E, winter night #040718 to #152050, snow #819EF6.
Rules (D, H by eye): the sky always lightens toward the horizon; dusk and night keep saturated hue (violet, navy),
never grey; warm emissive accents (windows, string lights, lamps) only at dusk and night; ground takes the
season's key hue. In game night (CF#883, NL#938, NL#961): blue violet grade with warm point lights and glow.

## 5. Characters and objects

### 5.1 Proportions (definitions first)
* H = figure height, top of silhouette (ears, hair, hats, horns included) to the lowest foot.
* Head fraction = (chin row, where the head meets neck, collar or body, minus silhouette top) / H.
* Core head = the same from the skull top, ears and horns excluded.
Readings are by eye on ruler montages (plus or minus 2% of H) or the automatic narrowest row below the widest head
row (vB `v_neck.py`, fails on neckless bodies).

| body type | head fraction (with ears) | source, confidence |
|---|---|---|
| NL player | 0.46 (0.44 to 0.47) | retail NL#940 0.46 [S, by eye, `synth/nl940_player6x.png`: hair top y 55, chin y 93, sole y 136]; art NL#19 to 28 0.47 [vD]; E3 2010 NL#947 0.44 [vB]. D's 0.49 and B's 0.50 to 0.55 REVISED. [Au] NL#940 at 6x: 0.44 (hair top) to 0.45 (ribbon top), chin row 91, sole 136: CONFIRMED low end. M to H |
| CF player | 0.57 (0.54 to 0.59) | art 0.57 [D, vD]; model renders 0.58 [B, vB]; in game CF#878 0.54 and 0.59 [vB]; [Au] CF#871 0.56 (hair top 58.8, chin 110.5, sole 151.3), CF#878 girl 0.56: CONFIRMED. H. CF players are visibly shorter legged than NL players |
| NL standard animal (cat, dog, pig, horse ...) | 0.50; core head 0.39 to 0.41 | B, vB, vD (17 std species read in art 0.51 and model 0.50). H |
| NL villager population | 0.53 to 0.55 (automatic over 333 renders 0.557) | vD. D's 0.57 REVISED |
| NL by species (one value per shared mesh, automatic) | mice 0.70, frogs 0.63, squirrels 0.61, hamsters 0.59, rabbits 0.58, cats 0.57, ducks and octopuses 0.55, cubs 0.54, birds and koalas 0.52, wolves, monkeys, rhinos 0.51, pigs 0.50, dogs, tigers, lions, alligators 0.48, hippos, deer 0.46, cows 0.45, elephants 0.44, goats 0.43, bulls 0.42, kangaroos 0.39 | vB. B's "cub 0.59, leg gap 0.04" row was mostly squirrels (REVISED: squirrel 0.61 gap 0.01; cub 0.54 gap 0.12; hamster 0.59). H |
| CF villagers | 0.60 (species 0.42 eagle to 0.68 mouse) | vD (D's 0.66 REVISED). M |
| special characters | 0.58 to 0.61 | B, D (UNVERIFIABLE by vD). M |

| further ratio | value | source |
|---|---|---|
| head width / core head height | 1.44 standard, 1.17 to 1.18 squirrels, cubs and birds, 1.30 big bodies | B, vB. M |
| head width / body width | 1.3 to 1.45 using full row width (arms included); 1.46 to 1.70 against the torso alone | D, vD |
| eye centre below core head top, fraction of head height | 0.33 standard, 0.47 squirrels, 0.42 birds, 0.67 players (hair takes the top third) | B (front views, n 54 to 57). M |
| eye height / head height; eye spacing / head width | 0.14 (cubs 0.18, birds 0.21); 0.38 (birds 0.57, big 0.32) | B. M |
| leg gap / H | 0.11 typical (0.09 to 0.16); 0 for hippos, elephants, rhinos, squirrels; kangaroos 0.04; cubs 0.12 | B, vB. H |
| arms | reach about hip height, never wider than the head (width ratio below chin over head 0.44 to 0.99) except hamsters 1.13, alligators 1.05, big bodies and winged birds | B, vB. M |
| figure | about 2 heads tall | all |

### 5.2 Geometry
* One body mesh per species (occasionally 2 or 3: anteaters, elephants, kangaroos, eagles); 316 of 333 NL renders
  fall in 39 silhouette groups (IoU at least 0.95, complete linkage), 17 singletons; members share proportions to
  0.01 H [B, vB]. Identity = texture (fur, markings, eyes, mouth, shirt) plus at most one or two accessory meshes
  (tuft, horns, hat). H for NL; CF cannot be tested.
* Round forms: visible silhouette corners turn about 25 to 30 deg, i.e. 12 to 16 segments per turn (B 12.9 to
  13.2 medians; vB's tangent method reads higher, 25 to 28, because ring vertices add small bends). [Au] own tracer
  plus Douglas Peucker at 0.4% of the figure height on the top 35% (heads) of 84 NL renders: median turn 25.3 deg, i.e.
  14.2 segments per turn: CONFIRMED. Use 12 to 16
  segments x 8 rings; CF facets are obvious at size (CF#409 head, hexagonal nose prism), NL looks rounder at its
  small size but measures the same [B, vB]. M to H.
* Smooth shading inside, faceted silhouette; hard edges only at real creases (muzzle boxes, bills, ear bases) [B, vB]. H.
* Parts: no fingers (nub or mitten hands), 5 to 6 sided arm tubes, stub legs, ear wedges or discs with painted
  inner colour, separate tail meshes, sawtooth shell coats on sheep, antlers; player hair is a separate shell with a
  zigzag fringe [B, vB by eye]. H.
* Triangle budget 500 to 800 per villager: arithmetic estimate only (13 x 8 head about 208, torso 64, limbs 144,
  extras about 100), true counts UNVERIFIABLE. L.
* Silhouette complexity of whole characters, Douglas Peucker corners at 1% of the diagonal: CORRECTED [Au] from "40 to
  60" (true for posed artwork only): NL front model renders median 35 (IQR 30 to 37, n 333, metrics; own tracer 33 on 84
  of them), CF model renders 39 to 55 (villagers 42, players 55, specials 39), artwork 39 to 52 (NL villagers 44, CF
  villagers 39, NL specials 46, CF specials 51); icons and objects 21 to 22. M.

### 5.3 Faces, outlines, surface
* Eyes, mouths, cheeks are texture decals (pixel identical silhouettes carry different faces) [B, vB]. H. Eye
  types: black disc with a white wedge glint, black oval in a white almond, coloured iris in a white almond with a
  thicker dark upper lid, closed arcs, sparkles [D]. Blush: soft salmon disc under the outer eye corner (#D7836E to
  #E08A76 on NL#30, not re measured). Mouth and detail lines = local fill at 40 to 60% value, never black [D].
* No ink outlines on characters, props or terrain in any game capture (CF#871, 878, 882, NL#940, 941 to 945, 947,
  949, 966) [A, B, C, D, all verifiers]. H.
* Surface: NL textures carry fine felt or plush grain, 2 to 4x CF (median absolute luminance minus 3x3 mean at a
  270 px figure: NL 1.44 to 1.74, CF villagers and specials 0.40 to 0.41) [B, vB]; NL grain confirmed in game on
  NL#966. CF fills are clean. H.
* Saturation of character renders: NL 0.44, CF 0.47 to 0.56 (mean HSV S) [vB].
* Props: fishing rod CF#447 is a 6 sided tapered tube in gold #C88838 / #D8B868 and navy #585888 / #181848 bands,
  white line #B8B8B8 to #F8F8F8, bobber red #C84828 top, yellow band, green bottom, 12 to 14 sided [B, vB]. Props
  bake 3 to 4 posterised shading bands plus a painted glint into the texture [B]. Use as a construction reference
  only, design your own colours.

## 6. Textures and formats

| asset | finding | source, confidence |
|---|---|---|
| CF clothing | 32x128 RGBA, fully opaque; at most 16 colours (median 16, mean 13.9, 268 of 346 exactly 16); 345 of 346 on the RGB565 grid (rounding expansion, not bit replication: only 38 fit that; [Au] recount 25 for bit replication, 345 for rounding); 334 need 6 bit green ([Au] 335); inferred CI4 with RGB565 palette (on disc format UNVERIFIABLE) | C, vC. H |
| CF clothing UV | four 32x32 blocks top to bottom: front, back, sleeve, sleeve (block 0 = 1 exactly in 148, 2 = 3 in 253); neckline at the top rows of front and back, cuffs at the bottom of each sleeve; 217 of 346 tile horizontally; sleeves often painted as ONE 32x64 panel with the motif crossing row 96 (51 textures; "only on uniforms" REJECTED) | C, vC. H |
| sleeve motif scale | drawn about 1.6x taller and 1.25x wider than on the torso (median run length ratios, 175 textures); C's "always 2x tall" REVISED | vC. M |
| palette use | dominant colour covers a median 50%; no dithering (2x2 checker share 0.0005, six deliberate weaves); per block colours 16, 14.5, 7, 7; stripes 3 colours, checks 8, everything else 16 (extra entries are antialias tones and 3 to 5 shade ramps, step 0.032 OKLab L); near black in only 66 of 346 | C, vC; [Au] recount: dominant share median 0.50, block medians 16, 14.5, 7, 7, near black (max channel below 40) 66, blocks 0 = 1 in 148, 2 = 3 in 253, all 346 are 32x128 opaque, max 16 colours, 268 exactly 16, mean 13.9: CONFIRMED. H |
| emblem outlines in textures | character emblems: 1 texel darker shade of the motif's own hue (V about 0.67 of fill); white motifs: dark shade of the GROUND hue; hearts, letters, numbers: no dark outline, only an antialias tone | vC (C's universal rule REVISED). M |
| CF UI icons | 64x64, 15 colours plus transparent, binary alpha in 34 of 36, alpha levels on the 3 bit grid (RGB5A3); outline 1 px hue matched dark (ring median #564229, V about 0.72 of the adjacent pixel); the khaki outer ring is background bleed from the olive pocket slot (#8A8D43), NOT a style rule (REJECTED) | C, vC; [Au] 36 of 36 are 64x64, 34 binary alpha, at most 15 colours; darkest ring / next ring inward V median 0.66 (IQR 0.58 to 0.78), ring colour median #534229; the outermost ring is brighter (1.33, the bleed): CONFIRMED. H |
| CF large icons | 96 to 102 px, 8 alpha levels in all 21 (RGB5A3), no dark rim (disc badges) | vC; [Au] 8 levels in 21 of 21, sizes 96 to 99 px in the 14 most common: CONFIRMED. H |
| NL item icons | 32x32, ETC1 plus 4 bit alpha (96 of 98 pass a strict grey axis block test; controls fail); 16 alpha levels, soft edges; 1 px outline at about 0.45 of the fill body (0.6 to 0.67 of the pixel just inside), hue within 3 deg (median), rim S 0.75 V 0.35, median #4C3719, near black on 2 to 4% of rim pixels; same designs as CF icons at half size | C, vC; [Au] 98 of 98 are 32x32 with every alpha value on the 17 step (4 bit) grid, 52 use all 16 levels; ring / body V median 0.45 CONFIRMED; ring / pixel just inside 0.56 (IQR 0.47 to 0.62), slightly below the v2 0.6 to 0.67. H |
| NL question icons, signs | resampled files (fail ETC1): do not read texel sizes from them | vC |
| icon fill | item bounding box about 75 to 80% of the cell (C's 80 to 90% REVISED); opaque area about 30% of the cell; specular blob about 3.5% of interior in 25 of 36 (CF), 60 of 98 (NL); no consistent light direction | vC. H |
| filtering | textures bilinear (soft texel blocks), no nearest stairs inside figures; in game geometry edges aliased in both games | B, vB, A. H |
| texel density | about 1.25 (NL) to 1.5 (CF) screen px per texel at gameplay distance (32 texels across the torso front); CF faces about 40 to 75 texels across in renders | C, B by eye, UNVERIFIABLE [vC, vB]. L |
| recommended sizes [S from the above] | clothing 32x128 strip per outfit; character body/head 128 (CF) to 256 (NL) square; props 64 to 128; UI icons author at 64 (CF look) or 32 (NL look) and display at 1.5x or 3x with Closest | estimate |
| RGB565 snap | round(k*255/31) for R and B, round(k*255/63) for G (`work_C/paint_recipes.py to565`) | C, vC. H |

## 7. Colour system

### 7.1 Distributions in scene captures (in game plus press, title screens and emote crops removed; C, vC reproduced exactly; H)

| per frame | CF (18) | NL (25) |
|---|---|---|
| saturation mean / median | 0.52 / 0.46 | 0.53 / 0.55 |
| value mean / median | 0.57 / 0.59 | 0.63 / 0.64 |
| neon (S and V above 0.85) mean / median | 1.1% / 0.7% | 1.6% / 0.1% |
| near black (V below 0.08) mean / median | 2.3% / 0.2% | 1.0% / 0.1% |
| near white (V above 0.92, S below 0.12) mean / median | 1.0% / 0.3% | 1.5% / 1.0% |
| pixels with S at least 0.8 | 24% | 16% |
| pixels with V at least 0.8 / below 0.3 | 19% / 16% | 28% / 10% |
| top hue families | orange 0.32, azure 0.22, cyan 0.11, blue 0.10, spring 0.09 | orange 0.18, spring 0.14, azure 0.11, blue 0.11, red 0.10, green 0.09 |

Rules: saturated but almost never neon; almost no pure black or white; orange (wood, dirt, sand, skin, fruit) is
the largest hue mass in every category. CF darker and more saturated; NL lighter with hue spread more evenly. Raw
albedo (textures 11.7%, artwork 13.4% near white means; medians 1.7% and 7.5%) carries far more white than rendered
scenes, so white albedo must render as light grey in shade [C, vC].

### 7.2 Master palette with roles (C, k = 28 weighted k means in OKLab over all categories; not re measured by vC; M)
White cloth / clouds #F2F3EC (rare in scenes); sand, cream, UI panel #DFCEBC; pale yellow #EEEA6A; gold, bells,
UI frame #E7BE57; light wood, dirt #C99A4F; wood, trunk #8B613E; dark wood, icon outline brown #622C20; khaki
#99835C; warm shadow #514D42; neutral shadow, dialogue box #323032; darkest accent (eyes, night, sparingly)
#111017; grass deep #2C7449; grass light #459B4F; NL lit foliage #7FC843; water mid #3C599A; light water #6B97C6;
pale cyan #90D0DF; CF saturated sky #223BDC; indigo night #30276D; violet accent #8356C5; cool grey stone
#6A717D; light grey paving #A5A5A3; red fruit, flowers #D43436; red shade #9B312F; orange #E4724A; pink cheeks
#EC9DA7; magenta accent #E063BB (never in scene captures). #9CE730 is the NL emote crop ground only.

### 7.3 Role colours (scene captures; sky, water, grass, sand re measured by vC; others C only)
See sections 4.1 to 4.3 for sky, ground, water. Further roles (C, M, one or few frames each): cloud CF #BFDEFF,
NL #B3C0CD; tree foliage CF #31761C (dark #1D4F10, light #589928), NL #4A9C52 (#287046, #82D362); wood CF
#C08E69 (#843D1F to #E7BC84), NL #E4A560 (#A1693B to #FFD391); player skin CF #F9B38B (#C58D77 to #FFBB93), NL
#FFD1A8 (#CE8F79 to #FFECC2); rock lit and shadow pair #CAAFA7 / #5E5A78 (CF), #D6C3BC / #535369 (NL) [vD].

### 7.4 UI colours (C, CF#872, 878, NL#948, 966; M)
CF pocket panel cream yellow #F1D679 frame, olive slot #898F49, tan #C59A5E; bells bubble #4EACEB / #ADD4F7; tab
bar sage #6D9893, tan #AD9263, ivory #D0CAAD, active tab orange #D79244; clock amber #C1880F, face #EAEAE4. NL
dialogue translucent near black brown #0F0B07 to #241810 with white text; name tags cream #FCFDC2, pink #FA88AE /
#FCBFD8.

## 8. 2D art, UI, typography (for thumbnails and on screen callouts; ORIGINAL designs only)

Never reuse the logo, clock crest, leaf badge with a bite, house shaped glyph tiles, the slab serif lettering, any
character, or franchise shirt motifs (CF#485, 631, 779, 783, 791 and similar) [D, vC].

* House style: a 3D toy diorama with a flat 2D graphic layer for branding and UI. About 688 "artwork" character
  images are CG renders of the game models (CF#22 to 309, NL#19 to 418) [D, vD]. Drawn 2D appears only in flat
  vector illustrations (NL#436, 451, 453, CF#341), CF leaf canopy frames, print and logos, and concept art. H.
* Lines: no outline on characters; a dark outline only on small UI icons (section 6) and on display lettering on a
  badge (near black #1F1F1D at about 0.45 of the stroke width; NL#0 also holds a second near black #0A0601, 4.6% of
  opaque pixels) [D, vD]. Sticker look: white die cut border 2.5% of figure height (NL#21; D's 2.8% REVISED) [vD].
* Title construction principle (NL#0, CF#0 measured): a physical object, here a die cut wooden plank (#8F6A36 /
  #8E6935 face, about 41% of opaque pixels; extruded side #523E1F to #554120 toward the lower right), letters as a
  warm yellow face (#E6B945, about 20% of opaque pixels, D's 22.5% REVISED) with a light top left bevel (#EFD18E)
  and a dark lower right drop; a second badge in a contrasting nature colour (#61BA46 with rim #55A23E, lemon
  #FFF33E letters, #FFFAC1 highlight, ochre bevel #A68537) [D, vD]. H. Apply the principle (object plus bevel plus
  drop, bouncing baseline, rounded slab serif or heavy rounded sans with an open licence) with your own shapes.
* Print language: CF cream page #FFF1D5 with tiny stars, white rounded display type on a coloured half disc
  header, fully rounded pills, 1 px leader lines ending in dots; NL kraft #D8B888 to #D8C898 or green #78B878 pages,
  rounded panels with a double border (#E8B838, #2888C8, #A8C848) [D, not re measured, M].
* In game UI: everything rounded, no sharp corners; CF translucent amber pocket bag with olive slot discs, round
  gold rimmed clock top left (x 54 to 182, y 28 to 139 of 832x456, plus date tag to y 180) [A, vA]; NL dialogue box
  dark translucent, bottom 43 to 45% of the top screen, striped cream border, pink rounded name tab top left, blue
  chevron "next" cue [A, vA, D].
* Key art composition families [D, vD]: (A) CF vignette: flat leaf canopy across the top, sky band from 17 to 21%
  down to 27 to 38%, skyline centred, two large characters in the lower third; (B) NL fisheye bowl: sky the top
  third (ends at 0.33 to 0.38 at the centre, 0.30 to 0.51 across the width), one hero tree centred on a planter,
  five characters in an arc in the lower 40%, foreground props cut by the frame, logo over sky never over faces;
  (C) flat print: 60 to 70% negative space, one tone silhouettes along the bottom, CF#341 sky #92CEE0 to #C0DFE4
  over a brown ground band #8A6C3F in the bottom 15 to 25%.
* Flat art grass is lime (#9CC948, hue 81, V 0.79, NL#436), which matches the RETAIL NL emote grass (median #92DB10,
  hue 81 to 82 [Au]), not the press build teal green [vD]. H.

## 9. Scene composition and cohesion (rules)

1. Camera looks down about 35 deg; the sky is a thin strip (CF gameplay 0 to 7% at the centre column; NL 13 to 18%
   at the centre, up to 30% at the sides). Gameplay sells the ground; key art sells sky and faces [A, vA, vD]. H.
2. Player centred horizontally, feet at 54 to 64% (CF) or 67 to 68% (NL) down; character 20 to 21% (CF) or 31 to
   34% (NL) of frame height [A, vA, Au]. H.
3. Horizon line is a row of tree trunks and crowns with sky in the gaps; far objects rise above the ground edge
   with their bases hidden by the bend [A, vA by eye]. H.
4. Three depth layers: a foreground occluder cutting the bottom edge (bushes CF#873, 888, NL#953, 956, 964), the
   player in the middle band, the trunk line and sky in the top 17 to 20% [A, vA]. H.
5. Density per outdoor frame: about 4 to 6 trees, 2 to 5 flower clumps, 1 to 2 rocks or shrubs, 1 building edge
   (estimate by eye; UNVERIFIABLE as numbers). Ground is never empty: scattered motif shapes and flowers [A, D]. M.
6. Scale relationships: NL street lamp 1.36x the player at the same depth (NL#961); CF cedar about 2.3x (by eye);
   flower clumps 0.4 to 0.5 P wide (by eye); museum tile 0.86 to 0.89 P [A, vA]. M.
7. Focal hierarchy: the centred player plus the saturated item in hand; one saturated accent (red fruit, red
   bobber, balloon) carries the focal point [A, D]. M.
8. Colour neighbours: CF sets cobalt sky against teal grass with warm orange trunks and a crisp horizon step; NL
   press builds dissolve the horizon in warm haze; retail NL uses deep blue sky over lighter lime grass [A, vA, S].
9. Absent on purpose: cast character shadows, outlines, DOF, motion blur, antialiasing (game), lens effects (except
   glow around lamps and windows), fog in CF, pure black, grime, sharp corners, threat [A, D]. H.
10. Why it feels cozy (synthesis, M): soft frontal light with violet, never black, shadows; 2 heads tall baby
   schema; felt, painted wood and plastic materials; saturated never neon colour; a world bounded by curves
   (rolling horizon, bowls, leaf frames); calm expressions; chores staged as celebrations.

## 10. City Folk vs New Leaf summary

| item | CF | NL |
|---|---|---|
| pitch, vFOV, focal (24 mm sensor height) | 35 deg, 22.5 deg, 60.3 mm | 35 deg, 28 deg, 48 mm |
| eye to player (interior) / camera height outdoors | 8.2 P / about 6.3 P | 4.3 P / about 3.6 P |
| player share of frame height / horizon row | 20 to 21% / 19.7% | 31 to 34% / 16.5 to 17% |
| bend (start under the camera; start at the player) | cylinder, R about 34 to 44 P; about 13 P (L) | cylinder, R about 31 P; about 13 P (L) |
| sky | cobalt #1A2AD0, thin lighter band at the horizon | E3 2010 cyan #45C5FF; E3 2011 and retail deep blue #2135E7 |
| grass | teal green #009758 | press #188050; retail lime #92DB10 (mottle #82CC09 to #A1E917) |
| haze | none | strong in press builds, toward #94CC78; retail ground and water only, toward #88BF68, far over near 2.15 (16.3, 16.17) |
| shadow side / lit, in game | about 0.5 to 0.6 | about 0.5 to 0.55 |
| rim | none confirmed | 1 to 2 px, plus 15 to 50% (head top strongest), pink lilac |
| blob shadow | violet tint (B multiplied least), core 0.3 to 0.6, about 0.4 P at 50% | R = G, B lifted, core 0.48, 0.38 P at 50%, sharper edge |
| player head fraction | 0.57 | 0.46 |
| character texture | clean flat fills, visible facets | felt grain 2 to 4x, rounder look |
| ground | hard decal shapes with dark teal outlines | soft edges, fine triangle print, seasonal motifs |
| water | deep, opaque, large caustic tile | lighter, swells, transparent, foam line |
| antialiasing | none, horizontal resample blur only | none, hard jaggies both ways |
| UI icons | 64 px CI4 RGB5A3, binary alpha | 32 px ETC1A4, soft alpha |
| artwork | deep promo shadows (0.24 to 0.31), darker silhouette edge, leaf vignette frames | lifted shadows, lilac rim, fisheye bowl key art |

## 11. Previous blueprint audit (PREVIOUS_BLUEPRINT.md, items 1 to 10)

The previous values came from 9 user images and the script `style_test_beach/beach_style_test.py` (player 1.03 m,
so 1 m is about 1 P there).

| # | previous claim | verdict | new value and evidence |
|---|---|---|---|
| 1 | LUT shadow ratio (0.550, 0.592, 0.747), highlight (1.265, 1.282, 1.076) relative to median, from a marlin model render; characters at half strength shadow tint | shadow CONFIRMED; highlight REVISED; half strength RETIRED | same metric on the same rock model in game [S `synth/lut_rock.py`]: NL shadow (0.535 to 0.550, 0.554 to 0.574, 0.647 to 0.660), CF (0.587, 0.645, 0.790): within 0.04 except a slightly smaller NL blue lift. Light end NL (1.46, 1.38, 1.20), CF (1.34, 1.31, 1.13), upper bounds (texture included). In game characters shade as deeply as rocks (hair p5/p95 0.56 NL vs rock 0.52 to 0.54 [vD]), so use the full LUT on characters; faces read because the key is frontal |
| 2 | day blob shadow neutral multiply 0.83 (one small JPEG, low) | RETIRED | NL core 0.48 (0.41 to 0.50, four retail frames), R and G equal, B lifted; CF violet (0.36, 0.36, 0.57) to (0.60, 0.59, 0.73) [A, vA] |
| 3a | day sky #215DF6 | REVISED | retail NL#940 #2135E7 (h 234) top, lower sky #4078DA to #6894D9 (NL#939) [S]; #215DF6 (h 223) sits between the two and is acceptable as a mid band |
| 3b | sand #CFBC9A | REVISED | corpus NL sand #FED08A (NL#949) and #CDA177 (NL#963) [vC], both more saturated (S 0.46 and 0.42 vs 0.26); no retail beach in the corpus, so the previous value cannot be confirmed |
| 3c | sea clusters #8DC2E7 / #6296C3 / #3A71AD | deep CONFIRMED, shallow REVISED | NL sea #3A75B4 to #3D7ABB; shallow shore #6BBEF0 [vC]; far sea #96B8F9 with haze [vA] |
| 3d | palm bands #D09A96 / #FEDAA8 | cannot confirm (no palms in the corpus captures) | retained as user measured, L |
| 3e | skin #E2BF9B | REVISED | NL player skin #FFD1A8 (shade #CE8F79, light #FFECC2) [C, M] |
| 3f | foliage #0E4D3E | REVISED | valid only as the dark tone; NL in game foliage median #4A9C52 (#287046 to #82D362) [C]; art leaf p5 #0C3D26, p50 #0E842E, p95 #4CE279 [vD] |
| 3g | rain, dusk, night tint multipliers relative to day sand | cannot confirm (not re measured) | in game references: CF night grass about 0.6 of day, characters (0.35, 0.37, 0.82) [A]; NL evening sky #7942FE, snow night #4E56B1 [vA] |
| 4 | shadows on whites shift blue violet; on coloured materials toward their own saturated hue | CONFIRMED with a CF caveat | rocks violet in both games [A, vA, vD]; NL whites lavender (render); CF whites neutral grey in renders (render only); warm fills 1 to 10 deg toward red with S +0.05 to +0.15 [vD] |
| 5 | no AA, binary alpha cutouts, bilinear low res textures, 400x240 native | CONFIRMED, one caveat | no MSAA in either game [A, vA]; NL model files binary alpha and in game edges aliased [vB]; bilinear [B, vB]; 400x240 [A]. UI icons are NOT binary in NL (16 alpha levels, ETC1A4) [vC] |
| 6 | camera pitch 40 deg, 56 mm on 36 mm wide sensor (about 36 x 22 deg), 6.5 m from the focus, player about 32% | pitch, lens, distance REVISED; player share CONFIRMED | pitch 35 deg [vA]; the 3DS frame is 5:3 like the script's 1200x720, so hFOV 45 deg, vFOV 28 deg: 43.5 mm on a 36 mm wide sensor (horizontal fit) [vA, S arithmetic]; eye to player about 5.2 P outdoors (4.3 P interior x 1.2), i.e. about 5.3 m in the script; player 31 to 34% [A, vA, retail NL#940] |
| 7 | rolling log cylinder, radius 11 m through the focus point (by eye) | CONFIRMED (shape), radius CONFIRMED within the low confidence band | cylinder, straight horizon [A, vA]; for a bend starting at the focus point the reviewed cameras give 11.7 to 14.3 P (NL) [S], i.e. 12 to 15 m in the script; tune to put the ground's end at 16.5% from the top |
| 8 | sun direction (0.15, minus 0.32, 0.94), LUT on N.L with emission shaders, no PBR | CONFIRMED in kind, lateral component REVISED | key from the camera side and above [A]; NL shows no left or right bias [vB], so set x near 0 (for example (0.0, minus 0.34, 0.94)); CF renders show a slight left bias (ratio 1.018) only. Elevation about 70 deg is not measurable (estimate). Smooth LUT on N.L CONFIRMED (no steps anywhere) |
| 9 | terrain blends sand, cliff, grass softly; caustic web; concentric shore foam bands | NL CONFIRMED, CF REVISED | NL edges soft (by eye) [A]; CF boundaries hard within 1 px with 2 to 3 px dark teal outlines [vA]; caustics both games; CF shore bands measured (light wet 16 px, dark wet 12 px, foam 3 to 4 px, 40 px ramp in 456 rows); NL a single white foam band of at least 5 px [A, vA] |
| 10 | chibi player head about 0.5 of body height (by eye) | REVISED | NL player 0.46 (0.44 to 0.47); CF player 0.57; NL standard animals 0.50 [S, vB, vD] |

## 12. Blender recipe

### 12.1 Constants (M = measured, with section; E = estimate or design choice; L = low confidence)

```python
import math

def focal_for_fov(fov_deg, sensor_mm):
    """Focal length that gives fov_deg across sensor_mm."""
    return 0.5 * sensor_mm / math.tan(math.radians(fov_deg) / 2)

P = 1.0   # player height (sole to hair top); every distance below is in P. The beach script uses P = 1.03 m.

NL = {
    # camera (section 1). Landscape 5:3 game frame values; see 12.2 for portrait.
    "pitch_deg": 35.0,                      # M  NL#950 parquet, vA
    "vfov_deg": 28.0,                       # M  25 to 31, vA (square plank assumption)
    "hfov_deg_5x3": 45.1,                   # M  derived: 2 atan(tan 14 deg x 5/3)
    "focal_mm_vfit24": focal_for_fov(28.0, 24.0),     # 48.1 mm, sensor fit VERTICAL, 24 mm sensor height
    "focal_mm_hfit36": focal_for_fov(45.1, 36.0),     # 43.3 mm, sensor fit HORIZONTAL, 36 mm sensor width
    "cam_height_P": 3.6,                    # M  outdoor (interior 3.0), vA x 1.2 outdoor factor
    "cam_ground_dist_to_feet_P": 4.1,       # M  outdoor (interior 3.45)
    "player_frame_share": 0.32,             # M  31 to 34 % of frame height (close view 0.45 to 0.52)
    "feet_row_from_top": 0.675,             # M  A
    "horizon_row_from_top": 0.165,          # M  NL#953, NL#946 (calibration target for the bend)
    # world bend: cylinder, axis left to right, bend starts at the focus point
    "bend_radius_P": 13.0,                  # L  11.7 to 14.4 P from the tangent model (section 1.3; Au 13.0 to 14.4); tune to horizon row
    # light (section 2)
    "sun_dir_to_sun": (0.0, -0.34, 0.94),   # E  camera at -Y looking +Y; frontal key, no lateral bias (vB); elevation E
    "lut_shadow_srgb": (0.54, 0.56, 0.65),  # M  rock dark 10 % / median, NL#946, 949 (synth/lut_rock.py)
    "lut_light_srgb": (1.46, 1.38, 1.20),   # M  upper bound (texture included)
    "shadow_to_lit_V": 0.52,                # M  in game p5/p95, 0.50 to 0.56 (vD)
    "rim": {"width_frac_of_height": 0.015, "gain": 0.22, "tint": "#C8B4E6"},  # M width and gain (vD, A; Au head top up to 0.50); tint E
    # gain is a DISPLAYED (sRGB) brightening: in linear light 1.22 is about 1.5x (section 3 audit)
    "haze": {"start_frac_of_ground": 0.5, "max_mix": 0.8, "colour": "#94CC78"},  # M press builds only; E mix curve
    # retail (16.3, 16.17): ground and water materials only, toward #88BF68, upper 100 ground rows, far / near 2.15
    # blob shadow decal (section 3)
    "blob": {"core_mul": 0.48, "core_mul_rgb": (0.48, 0.48, 0.62),  # M luminance core (vA, Au); rgb E: R = G
             # sRGB display ratios. Linear multiply that displays 0.48: about 0.19 (0.17 to 0.22). On lime grass B RISES
             # (Au): a multiply cannot do that, so blend toward a dark blue grey instead (blend colour E, tune on a frame).
             "core_mul_linear": 0.19,  # derived (section 3 audit)
             "width50_P": 0.38, "extent_P": 0.43, "edge_10_90_P": 0.09, "aspect": 1.35},  # M vA
    # palette (sections 4, 7). Retail look by default; "early" = E3 2010 press look.
    "sky_top": "#2135E7", "sky_low": "#4078DA", "sky_cloud_edge": "#6894D9",   # M NL#940, NL#939 (S)
    "sky_top_early": "#3BBBFF", "sky_horizon_early": "#50D0FF",               # M E3 2010
    "grass": "#92DB10", "grass_mottle_dark": "#82CC09", "grass_mottle_light": "#A1E917",  # M Au median, p25, p75 NL#941 to 945
    "grass_press": "#188050", "grass_dark_press": "#02613E",                   # M vC press; dark tone C (C_v1 grass table)
    "sand": "#FED08A", "sand_shade": "#CDA177",                               # M vC
    "sea": "#3D7ABB", "sea_shallow": "#6BBEF0", "sea_far": "#96B8F9", "foam": "#FFFFFF",  # M; foam E (white by eye)
    "cliff_lit": "#FFCEAE", "cliff_shade": "#BA8B8B",                          # M A
    "rock_lit": "#D6C3BC", "rock_shadow": "#535369",                           # M vD
    "foliage": "#4A9C52", "foliage_dark": "#287046", "foliage_light": "#82D362",  # M C
    "wood": "#E4A560", "wood_dark": "#A1693B", "wood_light": "#FFD391",       # M C
    "skin": "#FFD1A8", "skin_shade": "#CE8F79",                               # M C
    "evening_sky": "#7942FE", "snow_night": "#4E56B1",                        # M vA
    # textures (section 6)
    "tex_character": 256, "tex_clothing": (32, 128), "tex_prop": 128, "ui_icon_px": 32,  # E sizes; 32x128 M (CF)
    "fur_grain_target": 1.6,                # M median |L - 3x3 mean| at a 270 px figure (1.44 to 1.74)
    "ui_icon_alpha_levels": 16,             # M ETC1A4
    # models (section 5)
    "round_segments": 14, "round_rings": 8, "limb_sides": 6, "nose_prism_sides": 6,  # M 12 to 16 visible; E choice
    "head_frac_player": 0.46, "head_frac_animal": 0.50,  # M
    "head_w_over_h": 1.44, "eye_centre_from_top": 0.33, "eye_h": 0.14, "eye_spacing": 0.38,  # M B (animals)
    "leg_gap_frac": 0.11,                   # M
}

CF = {
    "pitch_deg": 35.0,                      # M  CF#876, 879 (Au 35.1); 37.6 in the CF#871 fishing view
    "vfov_deg": 22.5,                       # M
    "hfov_deg_native": 40.0,                # M  832x456 display frame (640 px native width)
    "focal_mm_vfit24": focal_for_fov(22.5, 24.0),     # 60.3 mm (v2 comment said 60.5: CORRECTED)
    "focal_mm_hfit36": focal_for_fov(40.0, 36.0),     # 49.5 mm
    "cam_height_P": 6.3, "cam_ground_dist_to_feet_P": 8.4,  # M Au outdoor refit (interior 5.2, 6.7); v2 5.9, 7.6
    "player_frame_share": 0.205,            # M Au 20 to 21 % outdoors (CF#871, 878); v2 0.215
    "feet_row_from_top": 0.59,              # M 54 to 64 %
    "horizon_row_from_top": 0.197,          # M
    "bend_radius_P": 13.0,                  # L 8 to 13 P; 12.9 at pitch 35 and horizon 0.197 (Au); v2 10.0 assumed pitch 36.2
    "sun_dir_to_sun": (0.05, -0.34, 0.94),  # E slight left bias only in renders
    "lut_shadow_srgb": (0.59, 0.65, 0.79),  # M CF#873 rock
    "lut_light_srgb": (1.34, 1.31, 1.13),   # M upper bound
    "shadow_to_lit_V": 0.55,                # M in game 0.49 hair, 0.59 rock (artwork imitation: 0.27)
    "rim": None,                            # M none confirmed
    "haze": None,                           # M none
    "blob": {"core_mul_rgb": (0.45, 0.45, 0.65),  # M between CF#877 (0.36, 0.36, 0.57) and CF#880 (0.60, 0.59, 0.73)
             "width50_P": 0.40, "extent_P": 0.65, "edge_10_90_P": 0.20, "aspect": 1.35},  # M; aspect E
    "sky_body": "#1A2AD0", "sky_zenith": "#040CBF", "sky_horizon": "#0D52E3",  # M vC, vA
    "sky_band_frac": 0.13,                  # M lighter band only about 60 of 456 rows above the horizon
    "sky_rain": "#748AA0", "sky_night": "#000C3B", "sky_night_horizon": "#002C6C",  # M
    "grass": "#009758", "grass_alt": "#0A8564", "grass_night": "#005C51",     # M vC, vA (night CF#883, A review)
    "ground_outline": "#00293E", "ground_outline_px_at_456": 2.5,             # M CF#888 (0,41,62)
    "dirt": "#C69342", "sand": "#E7CB8B",                                     # M
    "sea": "#25549A", "sea_bank": "#3275A7", "river": "#2886C5", "shallow": "#2A88AF",  # M
    "shore_bands_px_at_456": [("#E7CB8B", None), ("#E7D7A3", 16), ("#D3C397", 12), ("#ADDDE0", 4), ("ramp", 40)],  # M CF#871
    "caustic_period_px_at_832": (185, 227), "caustic_gain_p95": 0.18,          # M
    "cliff": "#CEA88A", "rock_lit": "#CAAFA7", "rock_shadow": "#5E5A78",       # M
    "skin": "#F9B38B", "skin_shade": "#C58D77", "wood": "#C08E69",            # M C
    "tex_character": 128, "tex_clothing": (32, 128), "clothing_max_colours": 16, "rgb565": True,  # M (128 E)
    "ui_icon_px": 64, "ui_icon_colours": 15, "ui_icon_alpha": "binary",      # M
    "fur_grain_target": 0.4,                # M clean fills
    "round_segments": 12, "round_rings": 8, "limb_sides": 6, "nose_prism_sides": 6,  # M visible facets
    "head_frac_player": 0.57, "head_frac_villager": 0.60,  # M (Au player 0.56 on CF#871, 878)
    "native_res": (640, 456), "display_res": (832, 456),   # M horizontal only resample
}

# scene envelope checks for any rendered frame (section 7.1): fractions of pixels
ENVELOPE = {"neon_max": 0.02, "near_black_max": 0.02, "near_white_max": 0.02}   # M scene capture means 1 to 2 %
```

### 12.2 EEVEE setup, step by step (Blender 5.2; follow CLAUDE.md pitfalls)
1. Units: model the player at 1.0 (or the script's 1.03 m) and derive every distance from P.
2. Camera: perspective, pitch 35 deg (rotation X 55 deg), no yaw. Landscape test frames: sensor fit VERTICAL with
   the vfit focal. Portrait shorts: sensor fit HORIZONTAL with the hfit focal (game width across 1080 px), then
   either accept the small player (11% CF, 19% NL of the frame width) or move to the NL close view for technique
   shots (section 1.4). Never use DOF or motion blur on the world (no game evidence); keep the repo rule on HUD.
3. Bend: Geometry Nodes on terrain and every placed object (shared group): z minus d^2/(2R), d = distance along the
   camera's horizontal forward axis from the focus point; rotate objects back by d/R about X; never bend sideways.
   Calibrate R so the ground ends at the horizon row (16.5% NL, 19.7% CF of a landscape frame) using
   `world_to_camera_view` on a far ground point, as the beach script's `--check` already does.
4. Light: no PBR. Emission shaders: albedo x LUT(N.L) where N.L uses the sun vector above, and the LUT is a
   ColorRamp with the shadow colour (sRGB ratio raised to 2.2) up to N.L about 0.08, white from about 0.38 to
   0.965, the light colour at 1.0 (stops as in the beach script, now with the new end values). Smooth, never
   CONSTANT interpolation. Disable sun shadows on characters (`light.use_shadow = False` or a shadow free LUT); keep
   the same LUT on characters (item 11.1).
5. Optional Gouraud look: store the LUT input per vertex (Store Named Attribute on the Point domain) so it
   interpolates across faces; untested here [vB].
6. NL rim: Layer Weight (Facing) through a Map Range so only the outer 1 to 2% of the figure height brightens,
   add the lilac tint at about plus 22% displayed luminance (about 1.5x in linear light); characters only. Check on a
   test frame that the 1 to 2 px ring reads 1.15 to 1.35x the interior (same ring method as vD `rim_hair.py`) and the
   head top up to about 1.5x (`audit/nl_rim.py`).
7. Blob shadows: a plane per character at the feet, slightly above the ground (bend applied), radial profile per
   section 3, material = Transparent BSDF whose colour is mix(white, blob tint, profile) (a true multiply, in LINEAR
   values: about 0.19 for a displayed 0.48 core; on lime grass add a dark blue grey blend so blue rises), render
   method Blended for static ground decals; if it smears under motion, switch to Dithered and fade by colour, not
   alpha (CLAUDE.md). Objects: smaller soft patches at trunk bases (ratio about 0.7).
8. Haze (NL, press and retail; CF off): Camera Data View Distance into Map Range (from about the middle of the visible
   ground to its far edge), smoothstep, then mix GROUND and WATER materials only toward the far colour: grass #88BF68
   (press #94CC78), cobbles and paths a lighter lavender grey (#C6C3D6, S#3152), sea #95B3F2. Never on objects:
   retail trunks, crowns and weeds at the horizon keep their colour (16.17). Anchor the distances to ground places,
   not to one framing, so a closer camera does not move the haze. Superseded the earlier "NL retail: off" (16.3).
9. Sky: CF: flat #1A2AD0 world with a horizon band only in the last 13% above the ground edge; NL retail: #2135E7
   to #4078DA. Standard view transform; apply the repo's camera ray compression trick only if the horizon clips.
10. Textures: paint with numpy (`work_C/paint_recipes.py` functions: to565, limit_palette, ramp, outline_colour,
   shirt_strip), add 1 to 2 antialias tones at motif edges before limiting to 16 colours [vC], paint sleeves as one
   32x64 panel. Image nodes Linear for 3D, Closest for UI pixel art. NL: add value noise so the grain metric
   lands near 1.6; CF: none.
11. Antialiasing look: clean is the default for shorts. For a native look render NL at 400 px wide (portrait
   400x711) with filter size 0.5 px and upscale nearest; CF at 640 px wide, then a horizontal only bilinear stretch
   by 1.3 (640 to 832 equivalent) before upscaling.
12. QA per frame: run the section 7.1 envelope (neon, near black, near white under 2%), the horizon row, the
   player share, and the blob core ratio on the rendered PNG.

## 13. Open questions and the evidence that would settle them

| question | why it matters | evidence that settles it |
|---|---|---|
| retail NL outdoor look: haze present? grass lime (#92DB10) or teal green? sky always deep blue? (RESOLVED in 16.2, 16.3, 16.17: haze yes, ground and water only; retail grass teal green #0D7541 family) | three of the strongest NL cues rest on press builds | 5 to 10 native 400x240 retail captures of open outdoor ground in summer and other seasons |
| bend radius and where the bend starts | sets how fast the ground drops; R spans 10 to 43 P across models | a retail capture with a straight road or fence receding over the horizon (track its screen curvature), or a frame pair with known walk distance |
| outdoor camera lens (no square grid outdoors) | interior grids only; 10% tile non squareness changes pitch 4 deg | outdoor captures with known square tiles (pattern paths) |
| CF rim term in game | NL has one; CF unconfirmed | a CF capture of a dark character against a dark background at native size |
| CF blob core (0.28 to 0.60 spread) | press images on pale paving disagree | CF in game captures of the player on plain grass |
| triangle counts, texture sizes of characters and terrain | budget and texel density are estimates | model files or wireframe captures; terrain texture rips |
| CF villager proportions per species (3/4 T pose renders only) | CF style animals | front views of CF villagers |
| does CF share meshes per species like NL | asset reuse rule for CF style | front renders of several CF villagers of one species |
| meaning of the lavender 1 px fringe on NL#947 (press) | rim versus compositing | same scene in a retail capture |
| exact sun elevation and azimuth | light vector is an estimate | a shiny sphere or long cast shadow of a static object in retail frames (no character cast shadows exist) |
| season to ground motif mapping in game | the motif table rests on key art | retail captures in each season |
| texel to screen ratio (1.25 to 1.5) | how soft textures look at gameplay distance | native captures with a known 32 texel shirt at gameplay distance, measured with an edge profile |

## 14. Audit log (third independent pass [Au], scripts and crops in `nook/audit/`)

Method summary. Camera: a third solver (`cam_widthline.py`, `nl950_cam.py`): on a floor of square tiles the
on screen tile width is a straight line in the row that reaches zero at the horizon row; with equal tile depth
spacing that fixes pitch and focal without a homography (principal point at the frame centre, square display
pixels). Frame shares and head fractions: ticked zoom crops read by eye (`ticks.py`, plus or minus 1 px). Colours:
channel medians inside own masks and boxes (`colours.py`). Shading, rim, blob: own masks and profiles
(`nl_emote.py`, `nl_blob2.py`, `nl_rim.py`, `lut_rock3.py`). Textures and icons: full recounts over every file
(`textures.py`, `icon_outline.py`). Silhouettes: own Moore tracer plus Douglas Peucker (`silhouette.py`).
Arithmetic: `arith.py`. Category of evidence in brackets: R retail in game, P press capture, M model render,
T raw texture or sprite, A arithmetic.

| # | number (v2) | verdict | v3 value, method, image IDs |
|---|---|---|---|
| 1 | CF pitch 35 deg | CONFIRMED | 35.1 deg, width line method, CF#876 [R] |
| 2 | NL pitch 35 deg | CONFIRMED | 35.2 deg (33.3 to 37.4 for plus or minus 5% tile widths), NL#950 [P] |
| 3 | CF vFOV 22.5 deg, hFOV 40 | CONFIRMED | 22.8 deg, hFOV 40.3, CF#876 [R] |
| 4 | NL vFOV 28 deg (25 to 31) | CONFIRMED (range) | third method 26.6 deg (24.8 to 28.7), NL#950 [P]; 28 kept, focal at 26.6 would be 50.8 mm |
| 5 | CF focal 60.5 mm (24 mm sensor, vertical fit) | CORRECTED | 60.3 mm: 12/tan(11.25 deg) = 60.33 [A]; NL 48.1, 43.3, CF 49.5 CONFIRMED |
| 6 | NL hFOV 45.1 (5:3), CF hFOV 40 (832x456) | CONFIRMED | 45.13 and 39.9 [A] |
| 7 | portrait (horizontal fit) vFOV 66 to 72 deg | CORRECTED | 65.8 (CF) to 72.7 (NL), written 66 to 73 [A]; vertical fit hFOV 12.8 and 16.0 CONFIRMED |
| 8 | CF player share outdoors 21 to 22% (0.215) | CORRECTED | 20 to 21% (0.205): CF#871 20.3%, CF#878 20.0, 20.3, 20.5% on the near row [R, P]; v2's own citations (20.8, 20 to 21) did not support 21 to 22 |
| 9 | CF player share interior 24% | CORRECTED | 25.2% (rows 175 to 290 of 456), CF#876 [R] |
| 10 | NL player share 31 to 34% | CONFIRMED | NL#946 31.7% [P], NL#940 33.0 to 33.8% [R]; NL#950 interior about 37% (38% v2) [P] |
| 11 | CF camera interior 5.4 / 6.9 P, eye 8.5 P | CORRECTED | 5.2 / 6.7 P, eye 8.2 P: share 0.252 plus feet row 0.636 solve, and independently width line height 6.0 tiles x 0.87 P per tile = 5.2 P, CF#876 [R, A] |
| 12 | CF camera outdoor 5.9 / 7.6 P | CORRECTED | 6.3 / 8.4 P from share 0.203 and feet row 0.59 at pitch 35, vFOV 22.5 [A on R, P values] |
| 13 | NL camera outdoor 3.6 / 4.1 P | CONFIRMED | predicts share 0.313 (measured 0.317); feet row 0.72 vs 0.675 measured, a refit gives 3.5 / 4.15 (within 3%) [A] |
| 14 | bend formula R = 2(H minus z0 tan alpha)/tan^2 alpha and NL R 14.3 P | CONFIRMED | derivation rechecked (tangency of the horizon ray); NL 14.4 P (v2 camera), 13.0 to 13.4 (refit); 31.6 under the camera [A] |
| 15 | CF bend constant 10 P | CORRECTED | 13 P: at the constants' pitch 35.0 and horizon 0.197, alpha 28.1 deg gives 12.9 P; 10.4 needs pitch 36.2 [A] |
| 16 | CF horizon row 19.4 to 20.7% | CONFIRMED | automatic last sky row: CF#870 19.7, 873 19.5, 878 21.1% [R, P] |
| 17 | NL horizon row 16.5% | CONFIRMED | 16.7 to 17.1% (40 to 41 of 240): NL#953, 949, 946 [P] |
| 18 | NL shadow to lit V 0.50 to 0.56 (constant 0.52) | CONFIRMED | hair p5/p95 median 0.53 over NL#941 to 945 [R] |
| 19 | CF shadow to lit V (hair 0.49) | CONFIRMED (band) | CF#878 hair 0.44, 0.44, 0.74 [P] |
| 20 | LUT light end NL (1.46, 1.38, 1.20), CF (1.34, 1.31, 1.13) | CONFIRMED | NL#946 (1.44, 1.36, 1.18), NL#949 (1.46, 1.35, 1.19), new NL#951 (1.45, 1.35, 1.17), CF#873 (1.30, 1.29, 1.12) [P, R] |
| 21 | LUT shadow end NL (0.54, 0.56, 0.65), CF (0.59, 0.65, 0.79) | CONFIRMED (mask sensitive) | eroded mask NL 0.64 to 0.65 / 0.74, NL#951 0.51 / 0.59, CF (0.62, 0.67, 0.81); v2 inside the spread [P, R] |
| 22 | rock lit / shadow hex CF #CAAFA7 / #5E5A78, NL #D6C3BC / #535369 | CONFIRMED | CF#873 #CAB0A9 / #5F5C7A; NL#946 #D7C5BE / #605F78; NL#951 #D8C5BD / #4C4E60 [R, P] |
| 23 | NL blob core 0.41 to 0.50 (use 0.48) | CONFIRMED | sole row profile 0.41 to 0.51 (NL#941, 942, 943, 944, 945) [R] |
| 24 | NL blob width at 50% 0.38 P | CONFIRMED | 42 to 46 px over a 109 px figure: 0.39 to 0.42 P [R] |
| 25 | NL blob tint as multiply (0.48, 0.48, 0.62) | CORRECTED (recipe) | R = G CONFIRMED (0.42 to 0.53); B rises 1.2 to 2.8x on lime grass, impossible for a multiply: blend toward a dark blue grey [R] |
| 26 | CF blob tint CF#877 (0.355, 0.356, 0.565) | CONFIRMED (kind) | (0.36, 0.40, 0.73), luminance 0.43; violet, B multiplied least; values sampling sensitive [P] |
| 27 | blob and rim values used directly as Blender multipliers | CORRECTED (colour space) | values are sRGB display ratios: linear multiply about 0.19 (0.17 to 0.22) for 0.48; rim plus 22% is about 1.52x linear [A] |
| 28 | NL rim plus 15 to 35%, 1 to 2 px | CORRECTED (range) | head top second pixel row 1.32 to 1.56x the hair interior with blue rising, third row 1.05 to 1.3: plus 15 to 50% [R] |
| 29 | NL player head fraction 0.46 | CONFIRMED | 0.44 to 0.45, NL#940 at 6x [R] |
| 30 | CF player head fraction 0.57 | CONFIRMED | 0.56 on CF#871 [R] and CF#878 girl [P] |
| 31 | CF sky body #1A2AD0, zenith #040CBF | CONFIRMED | rows 0 to 30: #1A22D2 to #1D31D1 (CF#870, 871, 873, 878); CF#882 #030CBE [R, P] |
| 32 | NL retail sky #2135E7 (NL#940), #242CE6 (NL#939) | CONFIRMED | #2139E8, #2323EF [R] |
| 33 | NL E3 2010 sky #45C5FF | CONFIRMED | #48CCF8 to #4DCDFF including the horizon band (NL#946, 947, 949, 951, 953) [P] |
| 34 | CF grass #009758 (CF#882), #009158, #0A8564 | CONFIRMED | #009658, #009158, #068464; CF#878 #009B59 [P, R] |
| 35 | NL retail grass #82CD0C | CORRECTED | median #92DB10, p25 #82CC09, p75 #A1E917, hue 81 to 82, V 0.86 (NL#941, 942, 944, 945) [R]; v2 value is the dark quartile |
| 36 | NL press grass #188050 / #147D4D | CONFIRMED | #127C4B, #188048, #1D8551 (NL#946, 953, 951) [P] |
| 37 | CF sea #25549A, CF sand #E7CB8B | CONFIRMED | #25539A, #E7CB8B (CF#871) [R] |
| 38 | NL sea #3A75B4 to #3D7ABB, NL sand #FED08A | CONFIRMED | #396FAD to #3A72AF (NL#957, 960, within 8 levels), sand #FED38C (NL#949) [P] |
| 39 | shirts 32x128, at most 16 colours, 268 exactly 16, mean 13.9, 345 on the RGB565 rounding grid, 334 need 6 bit green | CONFIRMED (335 need 6 bit) | full recount of 346 [T] |
| 40 | shirts: only 38 fit bit replication | CORRECTED | 25 of 346 under a strict bit replication test [T]; the conclusion (rounding expansion) stands |
| 41 | shirt palette rules: dominant share 0.50; block colours 16, 14.5, 7, 7; near black in 66; blocks 0 = 1 in 148, 2 = 3 in 253 | CONFIRMED | identical recount [T] |
| 42 | CF icons 64x64, 15 colours, binary alpha 34 of 36, outline V 0.72 | CONFIRMED | 36 of 36 at 64x64, 34 binary, max 15 colours; darkest ring / next ring 0.66 (IQR 0.58 to 0.78), colour #534229 [T] |
| 43 | CF large icons 8 alpha levels in 21 of 21 | CONFIRMED | recount [T] |
| 44 | NL icons 32x32, 16 alpha levels (4 bit), outline 0.45 of body | CONFIRMED | 98 of 98 on the 17 step grid; ring / body 0.45; ring / pixel inside 0.56 (v2 0.6 to 0.67, slightly lower) [T] |
| 45 | round forms 12 to 16 segments per turn | CONFIRMED | median head turn 25.3 deg = 14.2 segments (84 NL renders) [M] |
| 46 | whole character silhouettes 40 to 60 DP corners (1% diagonal) | CORRECTED | NL front model renders 35 (IQR 30 to 37, n 333; own tracer 33), CF model renders 39 to 55, artwork 39 to 52 [M] |

Totals: 46 numbers audited (the requested 30 plus 16 closely tied ones): 33 CONFIRMED (4 of them only within a band
or range: items 4, 19, 21, 26), 13 CORRECTED (items 5, 7, 8, 9, 11, 12, 15, 25, 27, 28, 35, 40, 46).

Consistency checks (section 12 against the body):
* After the edits above, the constants match the body for camera, share, bend, grass, rim, blob and textures; the
  block executes in plain Python (`focal_mm_vfit24` CF 60.33, NL 48.13; `focal_mm_hfit36` CF 49.45, NL 43.35).
  Body text gives NL horizontal fit focal 43.5 mm for hFOV 45.0 and the constant 43.3 mm for 45.1: same lens,
  rounding only.
* CF and NL presets do not mix measured values except `tex_clothing (32, 128)` in NL, which is the CF raw format
  (no NL textures in the corpus); its comment already says so. `grass_dark_press` (#02613E, C v1) and
  `grass_night` (#005C51, vA on CF#883) lacked provenance in v2; now commented.
* Confidence labels: no label claims more than its evidence after the edits; NL outdoor colour and haze items stay
  M because they rest on press builds; the retail grass correction rests on four retail emote captures (R).
* Writing rules: a scan of all prose outside code, hex codes and file names found no dash or hyphen used as
  punctuation or in compound words in v2; the v3 additions were scanned the same way.

## 15. Addendum after the first blueprint test scene (style_test_town)

New evidence and corrections found while building `style_test_town/town_river_test.py`:
* Retail NL grass (new source: a retail stream capture supplied by the user, game area 958x575, compressed, so M not
  H): near grass median #568F4F to #5D9D49 (hue 106 to 113, S 0.45 to 0.54, V 0.56 to 0.62), dark p10 #137434 to
  #287D36, lighter printed diamond patches p90 #9DCA64; far ground brightens to #9FCE60. This supersedes the lime
  #92DB10 as the default retail grass: the lime comes only from emote close up crops (NL#941 to 945) and, as a full
  frame ground, breaks the section 7.1 envelope (43% neon pixels in a test render versus under 2% in captures).
* Retail NL haze: the same capture shows the far ground drifting to #9FCE60, close to the press haze target #94CC78
  (section 2), so haze is likely present in retail too (M, one frame).
* LUT light end: the measured (1.46, 1.38, 1.20) is an upper bound that includes painted texture variation. Used as a
  lighting multiplier on top of painted textures it reads as shine; the test scene uses (1.10, 1.08, 1.04), an
  estimate matched by eye to NL#949 and NL#951.
* Surface detail target, measured on the 21 NL 400x240 captures with the section 7.1 method family: local grain
  (median of |L minus its 3x3 mean| over the frame) median 2.95 (p25 2.02, p75 3.75); edge density median 0.096;
  mean saturation 0.555; mean value 0.654. Flat colour materials and smooth sphere foliage scored 0.37 grain; painted
  textures on every material, leaf clump cards and printed grass raised it to 1.63 with saturation 0.561 and value
  0.669 inside the capture range. Every material needs a painted texture; no flat fills.
* Object shadows (second test pass): without them shrubs, flowers, trees and rocks read as pasted on. Measured on
  NL#946 against open grass #12804E: under the canopy #1C6840 to #185F3E (luminance 0.77 to 0.84; R rises because the
  grass R is near zero, G and B fall to 0.74 to 0.82), under a flower clump #185436 (0.68). Canopy shadow width about
  0.65 of the canopy width, centred, irregular soft edge (by eye). Implementation that reproduced the targets on the
  render (`town_river_test.py --render` QA): decal discs with the player blob blend colour, core alpha 0.72 (contact:
  flowers 0.32 P radius, rock 1.75 x its radius, trunk base 0.40 P) and 0.68 (canopy), noise wobble 0.10 to 0.16 on
  the radius; measured on the render: flower ground 0.68, canopy rim 0.85.
* Trees (by eye on NL#946, 953): fat flared trunk (base radius about 0.30 P, top 0.14 P), short (canopy starts at
  about 0.42 of the tree height), wide canopy of leaf clump cards; near trees have their crown cut by the top edge,
  horizon trees have their base hidden by the bend.
* Clouds: retail NL#939 shows thin low contrast wisps (see 4.1). Placed as unbent cards along camera rays beyond the
  bent ground so they only show inside the thin sky strip.
* Water (third test pass): the surface is a Blended mix of the water colour and transparency (alpha 0.45 at the banks,
  0.70 mid river, estimates tuned by eye), and everything under the line darkens in its own material by depth below
  the water line, stored before the bend: a 0.45 blend toward #A0A5AB at the line, then toward #3B6DAC over 0.25 P
  (NL#957). A dark fish silhouette (#0E2448, unlit, 0.06 P down) shows the translucency.
* Player head (retail NL#941, 942, 944, 945, emote crops, by eye plus pixel medians): skin median #FDDBB2; hair
  #7A5450; head wider than tall (radii used: width 0.235 P, height 0.205 P, depth 0.215 P); nose a small red orange
  nub #CF5846 just below the eye line, about 0.07 of the head width; ears round skin discs at the sides at eye to
  cheek height, about 0.19 of the head width, left visible below the hair; hair a separate shell (0.022 P off the
  head) covering top, back and sides down to the ear tops, with a zigzag fringe of about 6 teeth across the forehead;
  soft pink cheeks. Hair texture: low contrast mottle plus a sheen band, not stripes radiating from the crown (those
  read as a melon on a UV sphere).
* Lighting and shadows (fourth test pass, compared side by side with NL#946):
  * Key light as a camera side headlight along the reversed view direction, 35 deg above the ground (section 2:
    brightest band central on round forms, falloff toward every silhouette edge). The earlier near overhead key
    (elevation about 70 deg) left trunks and rocks flat lit.
  * Shadows MULTIPLY the ground (Transparent BSDF tint in Blended mode) so the painted grass runs through them, as in
    NL#946. Alpha blending toward a flat dark colour reads as a pasted sticker; keep it only for lime grass, where blue
    must rise. Displayed core multipliers used (converted to linear by the 2.2 power): player (0.54, 0.54, 0.68),
    contact (0.58, 0.58, 0.72), canopy (0.58, 0.60, 0.72); render QA: player R and G 0.44 and 0.49, flower ground 0.71,
    canopy rim 0.86 (targets 0.45, 0.68, 0.77 to 0.84). Profiles: player plateau 0.75 of the radius, contact 0.60,
    canopy 0.40 with a wobbly edge.
* Fish silhouettes (reference captures supplied by the user, by eye): a soft teardrop about 3 times as long as wide,
  round head (toward the bobber or the direction of travel), widest about 0.3 of the length from the head, straight
  taper to a pointed tail; dark (#0E2448) at about 0.6 opacity so the caustic lines show through. Draw it after the
  water: give the water object a far origin, because EEVEE sorts Blended objects by origin distance.
* Bark: painted swirl knots (concentric rings with a light centre, about 0.1 to 0.14 of the texture width) with the
  vertical grain streaks bending around them, plus darker crevices (NL#946, 953 by eye).
* Grain QA image: `style_test_town/grain_compare.py` puts a render beside reference captures with a heat map of
  |L minus its 3x3 mean| and the median. Last run, all resized to 400x240: ours 1.76 (flat pixels 23%), retail stream
  capture 1.64 (20%, softened by the stream upscale), NL#946 press 4.62 (10%). Flattest areas left: hair cap, skin,
  open water.

## 16. Screenshot corpus (16,939 images, seven sources)

Synthesis of six reviewers who viewed all 399 contact sheets ([R1] to [R6], findings in `shots/review/R*_findings.md`)
and three verifiers ([V1] checks R1 to R3, [V2] checks R4 and R5, [V3] checks R6 and reconciles grain). Rule applied:
where a verifier REVISED a value, the verifier's value is used; REJECTED claims are dropped or listed in 16.11 only.
[S] marks arithmetic done in this synthesis. Evidence class after each number: **LL** lossless RGB565 retail frame,
**Q80** retail 3DS album JPEG (IJG quality 80, 4:2:2), **J** other JPEG retail upload, **P** press capture, **Rn** model
render, **T** raw texture or sprite, **E** by eye. Confidence H, M, L. IDs are corpus items `S#index`
(`shots/corpus_items.json`); they are not the CF#/NL# IDs of sections 0 to 15. Lossy Fandom WEBP corpus copies were
never used for a colour value below (16.8). All values are sRGB display values unless marked linear; Blender multiplies
in linear light, so convert with the 2.2 power (section 3 audit).

### 16.0 Evidence base

| source | how obtained | items | what it holds |
|---|---|---|---|
| Nookipedia "City Folk screenshots" | category tree | 1,208 | CF retail frames 832x456 PNG and 640x480, look up crops, interiors, UI |
| Nookipedia "City Folk images" | category tree | 4,053 | CF raw textures (clothing, flooring, wallpaper, ground), icons, model viewer renders, publisher art, 1 PDF |
| Nookipedia "New Leaf screenshots" | category tree | 2,330 | NL retail frames, 721 of them lossless RGB565 PNG [V3], 1 MP4 |
| Fandom CF screenshots | MediaWiki API, original uploads | 1,420 | CF frames (many are Nookipedia twins) |
| Fandom NL screenshots, catalog, RV trees | MediaWiki API, original uploads | 7,396 | NL album JPEGs, catalog crops and renders, RV images |
| Nintendo Life | article galleries | 83 | NL press and review frames, JPEG |
| Neoseeker | the user's saved gallery page | 447 | NL retail JPEG frames, half of them 150x90 thumbnails |

Item counts are images analysed and sum to 16,937 [audit 16.13]: the catalogue holds 4,054 CF images records (the booklet
PDF excluded here), 2,331 NL screenshots records (the MP4 excluded) and 7,399 Fandom NL records (3 unobtainable).

* Counts [S from `corpus_totals.txt`, metrics]: 16,942 records; 3 unobtainable (Fandom "Blue Prom Dress 2/4, 3/4, 4/4",
  HTTP 400 on both the CDN and the original); 16,939 files obtained; 2 not images (an MP4 and the CF booklet PDF);
  16,937 images analysed, 5,480 Mpx in the first pass plus 1,718 Mpx for the 8,816 Fandom original uploads. Original
  formats (Fandom): JPEG 6,241, PNG 2,537, GIF 37, WEBP 1.
* Layouts of the first pass: 3DS top screen 5,024, Wii 832x456 412, Wii 640x480 382, 3DS bottom 60, both screens 24,
  Wii native 1, other sizes 11,034 (sum 16,937 [audit]).
* Kinds (reviewer counts by size rules plus eye; H for the big groups): NL 400x240 game frames 5,024 by layout, of which
  about 60% carry a dialogue balloon [R5, R6]; CF game frames about 1,890 [R1]; NL house interiors 508 and exterior crops
  448 [R4]; raw textures about 1,100 [R2, R3]; model viewer renders about 2,140 (CF) [R2, R3] and catalog crops and renders
  about 2,480 (NL) [R4, R5]; publisher and key art about 360 [R2]; UI screens several hundred; photos of a screen about 100;
  fan edits and banners about 150 [R1, R6].
* Retail versus press: press is a small minority: 10 CF E3 2008 grabs [R1], 21 NL E3 2010, E3 2011 and TGS frames [R4],
  part of the 83 Nintendo Life frames (some clean frames without HUD, cannot fully separate) [R3]. Everything else that is a
  game frame is a retail capture by players (HUD dates 2012 to 2024, Welcome amiibo content) (H). Nintendo Life items are
S#16412 to 16494 and Neoseeker S#16495 to 16941 [audit]; the Nintendo Life frames cited below (S#16417, 16419, 16441,
16443, 16455, 16472) are press or review captures whose retail status cannot be confirmed.
* Lossless retail frames: 737 at 400x240 pass an RGB565 test on every channel (721 Nookipedia PNG, 16 Fandom original
  PNG including S#9165 to 9177; 18 of the 737 are the top halves of 400x480 both screen captures [audit]) [V3; V2 counts 734 with a slightly different candidate rule]; plus 44 PNG at 1600x958
  that are emulator renders at 4x internal resolution with 16 bit output (RGB565 1.000, uniform 4x4 blocks only 0.03 to
  0.15, 958 rows, 2 short of 4x) [V2]. CF retail 832x456 PNGs are lossless but resampled from 640 px width (period 4.33 px,
  60 frames) [V1].
* Album JPEG signature: 2,898 of 3,292 native NL JPEG originals carry the full IJG quality 80 luma and chroma tables,
  2,875 of them 4:2:2 [V3, table match against Pillow references at every quality 50 to 100]; over all 400x240 items,
  about 3,440 quality 80 files (3,438 to 3,445 by selection rule) of 3,900 to 4,180 JPEGs [V3, audit]. Usable as an automatic retail filter. Neoseeker natives are near quality 99
  (4:2:2 in 175 of 214), Nintendo Life about 78 or 90 with 4:2:0 [R3, not reverified].
* Duplicates: 21 exact pixel duplicate pairs; 1,095 extra near duplicate copies by 16x16 average hash (coordinator); plus
  346 CF clothing "(CF).png" re encodes [R2, V1], 216 Neoseeker thumbnails [R3], 131 CF icon upscales [R3], 311 Fandom CF
  twins of Nookipedia files [R1]. Unusable for numbers: webcam and phone photos of a screen, desktop grabs, 3.5x and 2.0x
  resamples (683, colour only), thumbnails, and the lossy Fandom CDN WEBP copies (16.8).

### 16.1 Camera and framing by scene type

| scene type | game | measured | evidence |
|---|---|---|---|
| walking view, open ground | NL | ground starts at row 34 or 35 of 240 in all 8 frames: horizon 14.2 to 14.6% from the top, flat across columns | [V2, audit: 11 more frames at 33 to 35] S#1351, 10394, 11519, 2657, 2697, 10649, 10678, 13257; LL and Q80; H. R4's 13.3% (S#2692) was a building edge |
| walking view, player | NL | about 32% of frame height (76 to 77 px, hair top row 61 to 62, soles row 137 to 138, S#12899 [audit]; earlier 73 to 76 px); 31 to 33% by eye on S#1542, 1246, 2648; feet row varies 44 to 63% because the camera trails the player while walking and fishing | [V3, R4] Q80; M |
| walking view, centre column sky | NL | median 11 to 12.5% (p10 3.7%, p90 27%), a lower bound because trees and roofs cut it | [R3, R5, R6] J and Q80; M |
| talk (villager dialogue) view | NL | camera pushes in and lowers: horizon rows 43 to 46 (17.9 to 19.2%), a few at 39 to 40 or 47 [audit]; villager centre right, the player's head in the lower left foreground; balloon over rows about 160 to 230 | [V2] 12 frames; [R6] 1,100 frames E; H |
| catch view | NL | horizon rows 43 to 44 (18 to 19%); fixed close frontal camera, player 45 to 55% of the frame, head at rows 40 to 90, dialogue over the bottom 40% | [V2] 8 frames H; [R3, R5] E |
| tilt up at the town's north edge | NL | sky plus cloud fills 64% of the height, blue sky alone 50 to 52% (S#13187) | [V3] Q80; M |
| in front of a house | NL | ground ends 27 to 31%, sky 34 to 39%: same camera, the lot is a near horizon | [R4] S#1436, 1440, 1441; M |
| house exterior crops | NL | horizon at 0.787 of the crop height (p10 0.638, p90 0.795), house centred | [R4] 396 crops; H |
| swimming and diving | NL | low camera, head at the water line, sea about 80% of the frame, sky fraction 0.68 | [R3, R5] E and metric; M |
| villager house interior | NL | one fixed camera: player 30.8% of height (hair row 116 to sole row 189), feet at 78.8%, centred x 200; back wall top at rows 60 to 72; black outside the room box | [V2] per pixel median of 313 LL rooms, back wall rows [R4]; H |
| showroom, museum, RV | NL | steeper pitch about 45 to 55 deg, whole floor visible; RV rooms share one template with cab seats left | [R3, R4, R6] E; L |
| walking view, outdoors | CF | river fishing: player 22%, feet 68% (S#8242, WEBP copy, geometry only); sea fishing keeps the water centred, feet at 32% (S#230, L) | [R1] M |
| look up view (walking to the top edge) | CF | horizon 72.3 to 74.0% from the top where unobstructed (S#7710, 8279, 622 (a promo frame, P), 112, 8202, 8290, 8303, 8353 [audit]), 76.2% on S#8495 (rising sand mid frame, 73.3% at the bank), 62 to 71% where trees or houses stand on it | [V1] H; blueprint 73% CONFIRMED |
| house look up crops 420x308 | CF | sky ends 26 to 65% (houses and cliffs cut it) | [R1] H |
| villager house interior | CF | player 23.0 to 25.4% by hairstyle (105 and 116 px of 456 [audit]), soles at 61.6 to 61.8% (S#433, 541); black void median 8.5% of the frame (p10 2.8, p90 9.3), left 10% of columns 14.3% | [R1, V1] 217 frames; H for void, M for feet |
| caught item pose | CF | close up, head about one fifth of the frame, bust centred, dialogue over the lower 45% | [R1] E |
| museum | CF | steeper and higher view, about 40 to 50 deg by eye | [R1] L |

Lens in every retail frame of both games: no DOF, no motion blur, no yaw or roll, no antialiasing (1 px stair steps even
in the 4x emulator renders) [all reviewers, E; H]. One CF radial zoom blur series (S#8118 to 8121) also blurs the HUD,
so it is a transition, not a lens rule [R1, L].

Bend consequence [S arithmetic, L]: with the section 12.1 NL camera (pitch 35, vFOV 28, height 3.6 P, feet distance
4.1 P) the retail walking horizon 0.144 puts the horizon ray 24.9 deg below horizontal, so R = 2(H minus z0 tan a)/tan^2 a
= 15.7 P for a bend that starts at the player (14.5 P with the 3.5 / 4.15 refit), against 14.4 P for the press 0.165.
The camera itself was solved on press frames; a retail grid solve is still missing (16.12).

### 16.2 Sky by time of day, season and weather

**New Leaf, clear day (retail).** The zenith is one exact colour: #2120E7 on lossless frames (modal top colour on 23 raw
frames [V1], 26 of 45 lossless outdoor sky frames [V2]) and #2121E5 on album JPEGs (19,533 top row samples over 181
frames [R5]; 72 of 134 frames [V3]). H. What the walking camera shows is mostly the lighter part of a dome gradient:

| view | top edge of the strip | toward the ground | evidence |
|---|---|---|---|
| walking view (18 of 23 frames) | #2040F0 to #2050ED at row 0 | last sky rows #2166FA to #2172FD | [V2] LL and Q80; H |
| walking view, deep minority (5 of 23: town hall and event plazas) | #2121E5 | lightens late | [V2]; cause (camera or place) cannot be confirmed; clock time does not separate them |
| profile over 100 rows (S#10406) | #2121E5 from 100 rows above the horizon | #2126ED at 80, #213FEE at 60, #215AF6 at 40, peak #2169FC at 15 to 20 rows, small dip #2061EC in the last row | [R5, V2 within 7 levels]; H |
| deep lossless plazas (S#1373, 1378, 3152, 3062, 3061, 9171) | #2120E7 to 30 rows above the horizon | #3951EF at 5 rows, #5A65F7 at 1 row: a LILAC light, not cyan | [V2] LL; H |
| talk and catch views | #2121E5 | #2130EE to #2145F2 | [V2] about 30 frames; H |
| 60 row band average | #2341E7 (hue mask, 508 frames) to #244DE5 (RGB mask, 538 frames) | | [R6, V3] Q80; a mask dependent average, not a fill |
| variants | #2938E7 to #293CE7 (all Harvest Festival frames, Caught Bee: 7 LL frames); Tortimer Island #2145F7 to #2149F7; #005DC6 to #005DCE on about 45 LL frames with a blue top row is NOT sky: it is the title bar of the in game fish and bug encyclopedia (coordinator check, 40 frames viewed, e.g. NL Encyclopedia Angelfish); exclude those frames from sky statistics | | [V1, V2, V3]; M |
| look up over Main Street roofs (S#10396) | #428ADE to #488BDA in rows 10 to 60 | | [R5] J; M |

**New Leaf clouds.** Most walking frames show no cloud (by eye [R6]). Retail clouds exist in three forms: flat horizontal
puffs, median #949EF7 with bright tops to #FFF3CE, 3% of the sky (S#1373, LL) [R4, M]; towering cumulus with soft painted
edges plus thin wisps on S#13187 (Q80): 30.4% of rows 0 to 139, median #CCE1FE, p10 #A6C5FC, p90 #DDEAFD, maximum
(232, 250, 255), never pure white, sky beside them #212CEB [V3, H for the frame]; thin wisps (NL#939, section 4.1).
Low camera frames show soft cloud bands #D4CCC6, 15% of the sky (S#16443, Nintendo Life, likely press) [R3, L].

**New Leaf night.** The default night zenith is #01215A (modal top colour on 29 of 114 dark uniform top frames: S#1498,
1585, 3081, 3110, 3205, 9062, 10170, 11554, 13698 and others), then #072559 (7 frames), and a violet family #323663,
#403D6A, #424169, #4A426B (about 20 frames) [V2, H]. It is FLAT: #00205A in every 10 row band from 0 to 79 on S#12601
(an 807x484 PNG, about 2x, not a JPEG) and #01215A to row 49 on S#13760 (Q80) [V3, audit, H]. One violet frame adds a band #493E69 to #53386B over the last 20 rows
(S#1386) [R4, M]. Stars are single white pixels; the moon is an emissive disc with a halo: S#12601 core #F0F3C6, radius
about 10 px, halo #DBE8C4 at 8 px, #44658D at 12 px, sky by 20 px, in that frame's 807 px units (about half at 400x240) [R6, M]; a full moon frame shows a disc about 90 px across
(S#10672, tilted night camera, E) [R5, L].

**New Leaf dusk and sunset** (one retail frame each, M; the families are reliable, the hex values are per frame): early
evening teal grey #658B9A, #A19278, #C79A64 top to horizon (S#16455) [R3] or light blue #4E9FCC (S#9613, 5:09 PM) [R5];
lavender #84667A to #7A617A (S#10731) [R5], #403D6A to #603563 (S#12331) [R6]; magenta #6E3F94 (S#16595) [R3], land sunset
#594D71 with #6D4E6D to #805367 in the last 20 rows (S#1597) [R4]; sunset with a warm glow #7D617B to #C76153 (S#12510)
[R6], #7A4148 top, #9B554B a quarter down (S#1339) [R4], glow #D08F49 (S#10400) [R5]; evening 6:53 PM #4C466B over olive
grass #4A5012 (S#1404) [R4]. Ground and water take the tint: cobble #7B6781, ground #B57E60 or #634451, seas #473D51,
#AE5C5B, #322841, #4F383F.

**New Leaf weather.** Rain sky #7C8EA4 (S 0.25, S#16496) [R3], #7B8CA3 (S#10603, 11:38 AM) [R5], #364E69 (darker,
S#1453) [R4]; overcast snow sky #7B8DA3 (S#1416) [R4]; rain is thin light streaks, no fog [R4, R5, E]. Rainbow after rain,
S#9226 column x 110 outside in: #6940BA, #C74D5A, #DA8145, #DBAD58, #D6CD58, #ADD261, #5BC764, #0DB175, #0FABBA,
#0EA9EB, then sky #146BF6: about 7 smooth bands over 40 px with sky at both edges [R5, M]. Aurora (winter nights only):
curtains #6AEFFE, #D0EA87, #05B380 over navy #002764 to #0A406F (S#1218) [R4, M]. Press E3 2010 cyan #3DBDFF to
#45C5FF stays a press look (section 4.1) [R4].

**City Folk, clear day (retail).** Gameplay frames show almost no sky, so the values come from look up frames:

| set | body | top rows | horizon end | evidence |
|---|---|---|---|---|
| house look ups, mostly 420x308 crops (S#316 to 411, 82 frames with sky; 75 of the items are 420x308 [audit]) | #1714C9 (p10 #1713C8, p90 #1816CA) | #1511C4 (#1410C3 to #1512C6) | | [V1] H |
| 640x480 look ups (12 frames: S#112 to 114, 8036, 8202, 8290, 8303, 8321, 8353, 8378, 8407, 8787) | #1210BE to #1211BF | #0E0EB9 to #100EBA | #1A5CDC to #1B6CE2 (S#112 #1A63E0, 7710 #216AD6, 8495 #1B6CE2, 8279 #1B5FDF) | [V1] H |
| promo press kit S#622 | #0510C3 | #020CBE | #0C5CE6 | [V1] P |

The look up ramp occupies 93 to 134 of 480 rows above the horizon (19.4 to 27.9%, median 24.3%, 16 frames) [V1]; the
blueprint's 13% band (section 4.1) describes the press gameplay frames. **CF clouds are common in retail**: cores (min
channel over 190) #C1DCFF class, all cloud pixels with soft edges #B1CFFF to #C3DFFF; in the top 20% of a look up frame
they cover a median 29% (p10 19, p90 39) with soft edges, 18% for cores only: use 15 to 30% [V1, 79 frames, H]. A rainbow
appears on S#7710 [R1].

**City Folk other times** (retail): early night violet bands #312848, #352D4D, #383153, #413357 top to horizon (S#149,
HUD about 20:00) [V1]; late night #000B39 (S#8426, matches blueprint #000C3B) [V1]; dusk body #513C5A, lit clouds bright
#DDA495, all cloud #C88E8F (S#253) [V1]; rain #646E86 (dark #262B41, light #808BA2, S#158) [V1]. R1's overcast snow,
aurora and dusk horizon cloud values came from Fandom WEBP copies (S#7811, 8401, 7864) and are not used.

### 16.3 Ground: grass, motifs, sand, snow, dirt, borders, haze

**New Leaf grass, near band (rows 170 to 236, below the haze) [V2 unless marked; season from the HUD date].**

| condition | colour (hue) | IDs, class, confidence |
|---|---|---|
| spring and summer town | #026637 (152), #107442 (150; #1E7542 in V2, not reproduced [audit]), #107641 (149), #0C6C3A (149), #0D7541 (150), #06723F (152) | S#1246, 1245, 1341, 9788, 13257, 10649; LL and Q80; H |
| same, larger samples | #12723F near band over 22 frames [V1]; #14713E over 27 frames [R3] | J; H |
| late winter, no snow (25 Feb) | #5A8548 (102), #5B8D3B (97), #659844 (96): hue band 90 to 102 | S#1433, 1591, 1351; M |
| late autumn, early December | #7B673A (42), #8F631B (37, 27 Nov) brown leaf litter | S#1406, 10421; M |
| other autumn ground | #755150 (S#16419) [R3]; #745F40 golden, p90 #937A20 (S#12613) [R6]; orange #9A5937 (S#13255) [R6] | one frame each; M |
| Tortimer Island | #187939 to #318239 (126 to 140) | S#2652, 2657, 2693, 2694, 1542; LL; H |
| Main Street strips | #5A9E5A to #63A25A (112 to 120) | S#1373, 1375, 1379; LL; H |
| catch frames (raw, rows 0 to 150) | #42A04A (p25 #296929, p75 #7BB66B) | 44 LL frames [R3]; M |
| emote backdrop only | lime #94DD0F (p25 #83CD09, p75 #A1EA16), streaky, no motif | 40 crops [R3]; H, not a town grass |
| per frame spread, mixed dates | median of medians #2D7942 (p10 #13652D, p90 #559A54), hue 99 to 152 | 97 Q80 frames [R6]; M |

The section 15 stream value #568F4F to #5D9D49 matches Main Street and the light end of the spread, not the town default.

**New Leaf ground motifs (retail, by eye, M):** round dots in spring and summer (S#1341, 1246, 1351), leaf or star decals
in late winter (S#1433), brown stars in March (S#1221) [R4]; darker diamond or triangle print p10 #005A3D on #06663B
(S#12784), triangles on autumn ground (S#13255), worn bare patches #877A5F (S#12672) [R6]; round or hex patches in rain
(S#16650) [R3]. Catalog plinths show a large two green checker, #487D25 (45%) and #247149 (34%) (S#16385) [R5]: a
candidate grass base pattern, L.

**New Leaf retail haze: CONFIRMED, H.** On 22 retail frames the ground brightens monotonically over the 100 rows below
the horizon (about the upper half of the 205 visible ground rows [S]); rows 136 to 156 are still only 1.06 of the near band
[V2]. Far over near luminance: median 2.15 (range 1.61 to 3.19, 17 frames) with fixed 20 row bands [V2]; median 1.86 (p25
1.41, p75 2.27) with bands relative to the ground extent [V1]. Saturation falls from 0.81 to 0.97 near to 0.41 to 0.54 far
[V1]; R rises most (about 5 to 180), then G (115 to 220), B least (60 to 105) [V2]. Far band colours: #B5E36B (S#2657),
#99CB65 (1246), #80BD65 (1341), #D2EC6E (1351), #ACD76C (9788), #ABD570 (13257), #93C76D (10649), #B8E168 (2697), #BDE56B
(10507), #7FB85F (9086) [V2]; median #88BF68 against near #12723F [V1]. The target is a warm lime close to the press
#94CC78 to #AAE677, reached by a MIX toward it (desaturation proves it is not an additive brightening) [V1]. Haze also
covers the far sea: #95B3F2 band on retail S#16577 [R3].

**City Folk grass (retail).** Day #0A8C54 (IQR #078243 to #2B955D, hue 154, 240 frames) [V1; R1 #098858 on 154];
print p10 #007030, p90 #4AAB73 [R1]; night #0D5349 (hue 172, G 0.59 and B 0.87 of day, 23 frames) [V1]; dusk #0F5644
(S#253) [R1, L]; rain #275238 (hue 143, S#158) [V1, L]. No haze measured in retail CF (blueprint section 2 stands).

**City Folk ground textures (raw, T, H).** S#3573 to 3575 are 133x133 swatches: a 2 px black border, a 1 px divider and
four 64x64 quadrants per motif family, one palette per season, each with exactly 3 motif colours plus 140 to 201
antialias tones [R3, V1]:

| quadrant | base | motif colours |
|---|---|---|
| summer | #009059 | #60AF17, #02AE58, #0AA77B |
| second green (season not identified) | #008164 | #549667, #2AA755, #0A9A8F |
| snow | #C0D7EC | #F8F9F6, #E6F2F5, #E0E7F2 |
| dead grass | #97776C | #BF9A73, #B88D72, #AA7F89 (a mauve fourth tone) |

The three families are squares (S#3573), triangles (S#3574) and circles (S#3575), shapes 4 to 16 px in a 64 px tile,
randomly rotated; the snow quadrant swaps the shape to rounded squares, five point stars and hexagons respectively; the
motif layout is identical across the four quadrants of a family (palette swap only) [V1]. Retail towns use ONE family each:
squares (S#1167, 122, 1129, 8290, which read as rhombi in perspective), triangles (S#8506), circles (S#96) [V1]. The
retail snow star print (S#111) is the triangle town's snow. Wendell's grass S#3907 is the triangle town at texel level:
#00945A 63.4%, #63AD10 17.5%, #00AD5A 12.9%, #08A57B 6.2% [V1 exact count]. Grass blade cards (S#3840, 3866): 31x57,
two curved tapering blades, an 8 colour ramp from dark base to light tip [R3].

**Dirt and paths.** CF retail day dirt #C79748 to #C8903F (S#1129, 8290, 7710, 8495) [V1]; Wendell dirt S#3908 #CEA54A
63.8%, #BD944A 26.4%, #D68C31 9.9% [V1]; cobble S#3930 stones #94B5C5 / #7B8C94, grout #5A5252 [R2]. NL river bank
dirt #946D31 (p10 #7B5931, p90 #AD8E31, S#9176, LL) [R5, M]; NL plazas: cobble #AAA9B9 (autumn S#10421) [R5], #8390BC
(rain S#16650) [R3], dirt plaza #9D6223 (S#13502) [R6].

**Borders.** CF retail grass to dirt: a darkened lip on the DIRT side, dark brown #7E5018 (S#1129), olive #665824 to
#7D7945 (S#8290, 8495), dark teal #20514B only on S#7710 [V1]. The blueprint's teal outline (0,41,62) came from a press
frame. NL edges stay soft (by eye, all reviewers).

**Sand.** CF shore strip #E7CA8B on retail S#1136 (the blueprint #E7CB8B); inland beach ground is the dirt palette #CEA64A
with orange hexagon pebbles [V1; R1's proposed change REJECTED]. NL retail dry sand: island and beach #EFB67B (S#1539, LL)
[V2], #ECBB84 (S#16659) [V1], #F4BB8C, #E8B982, #F5A85D (S#12286, 13496, 13255) [V3], lossless beach #CEA673 (S#9175) [V2];
town beach #CA9A68 to #DDA374 (S#16577) [V1]. Wet band luminance 0.62 to 0.71 of dry (#B08460, #9B7760) [V3, REVISED from
0.70 to 0.74]. Printed shell decals (E) [R5].

**Snow (never neutral white).** NL day #A1A6E0 (p10 #969ED7, p90 #B1B3EA, S#3087) [V1], #8E90BA (S#1242) [R4], #E7D7DA with
warm highlights #FFEFEB (S#13855) and #B6AECF (S#12176) [V3], pink lilac #C499BC at 5:33 PM (S#10076) [R5]; star or hex
cobble print (E). NL night #4E56B1 (S#16417, equals NL#961) [V1], #5989EE (S#13760) [V3], #5257BD to #7977C7 [R4]. CF
snow #BDD3E6, white star print #F0F6F4, shade #A1BDC6 (S#111) [V1, R1]; snowman #9EBBF3 centre to #849FD5 edge [R1].

**Cliffs.** CF summer wall median #86645D (p10 #54243B, p90 #D7BF97), cream band #D9C399, crevices #4E1F36 (S#48); winter
#5E5E79 (S#8511) [V1]; faceted columns with a zigzag grass overhang [R1, E]. NL salmon lit #FCC599 (p10 #D1988D, p90
#FFE7B3, S#1242) [R4], #FAB7A0 (S#12787) [R6]; mauve pink slabs #B79EB1, lit top #FADCC6 (S#16472, 3087) [R3]; rain mauve
#A78DA6 (S#1453) [R4]; horizontal strata, irregular slabs, a grass lip (E).

### 16.4 Water: sea, river, shore, foam, wave lines, fish shadows

* NL sea (retail): near #3367A0, mid #3175B6, far haze band #95B3F2 (S#16577) [R3]; lossless #3169AD (p10 #2965AD, p90
  #4A79AD, S#9174) [R5]; #316DAD, #3975BD, #3B84C8 (S#2646, 2648, 1337) [R4]; with crests #4983C2, #4C8BC9, #3270B8,
  troughs to crests #1770C2 to #8FADD1 (S#12286, 13496, 13255) [V3]. Use median #3A76C4 [R6, V3]: section 4.3 CONFIRMED.
  Island sea #205AF5 (S#16659) [R3]. Shallow band #5AA2DD (S#10230) [R5]; water over sand near the shore #A5C3E7 to
  #C6D3E7 [R4]. Translucency CONFIRMED in retail: swimmers and sand show through (S#2646, 1539) [R4, M].
* NL river #316BAF (S#3087) [R3], #2A68AA, fishing frames #3663A3 to #3C64A2 [R6]. Wave and caustic lines +42% at
  p95 and +65% at p99 luminance on the sea, +28% and +47% on rivers [R3, M], +21 to 41% at p95 [R4]; cell spacing 30 to 55
  px (noisy) [R4, L], so the press period 115 to 120 px cannot be confirmed; pool caustic reference base #0378D2, lines
  #5BD5FF over about 21% (S#16316) [R5, Rn].
* NL shoreline (S#12286, 13496): dry sand, wet band, foam line #ECF2F8 / #DBDDF6 [V3], sea with lighter crest scallops
  (#FFEFC6 [R4]); waves are wobbly lines parallel to the shore (E). Time of day: dusk sea #3C5596 with violet wet sand
  #86779D [R5], dusk pond mauve #70455F [R6]; night sea #0E1D53 to #111F53 [R3, R6].
* CF water: plunge pool #213B8B (S#285); sea #1C5196 darkening toward the camera to #183B8B, shallow wash #4681AD
  to #3C88B6, evening wet sand #7D8390 (S#230); waterfall sheet #A3E0DC, foam #DAF5F2 (S#285) [R1, M].
* **Fish shadows, NL retail Q80 (S#12898 to 12902), [V3] values:**

| class | length x width px (threshold 0.85 of water) | bobber px | length in bobbers | length in P (per frame bobber scale) |
|---|---|---|---|---|
| tiny | 10.4 x 6.5 | 9x9 | 1.2 | 0.16 |
| small | 22.8 x 7.3 | 10x8 | 2.3 | 0.30 |
| medium | 24.0 x 10.1 (points at the camera, foreshortened) | 10x9 | 2.4 or more | 0.31 or more |
| large | 38.0 x 11.2 | 8x9 | 4.8 | 0.63 |
| very large | 44.0 x 11.3 | 7x7 | 6.3 | 0.82 |

  P = 76 to 77 px = 7.6 to 7.7 bobbers on S#12899 [audit; V3 had 73 to 76 px = 7.3], and the P column uses 7.65 [S]. Edge CRISP: plateau 0.55 to 0.63 of the water, boundary pixel 0.70 to
  0.75, one pixel outside 0.91 to 0.95, two outside 0.95 to 1.05 (a 1 to 2 px transition even after JPEG 4:2:2). Luminance
  median 0.60 to 0.66 of the water, darkest 10% 0.52 to 0.58. Shader multiply: use the per channel MEDIAN ratio (0.45 to
  0.61, 0.61 to 0.67, 0.66 to 0.73), about (0.50, 0.63, 0.68) displayed, i.e. (0.22, 0.36, 0.43) linear [S]; R6's p10
  (0.35, 0.57, 0.62) is too dark. Red falls most, so the shadow reads teal navy (median #1B3D6A on water #3863A2 [R6]).
  One end tapers over about 5 px [R6, E]. CF fish shadows (S#74, 75): soft dark teardrops about 3:1 (59x20 px), core
  #07131E on #253B7A (darkest 3% at 0.46) and #0E2E3A on #285799 (0.73) [R1, unverified, M].

### 16.5 Lighting

* Key and shading: camera side headlight, smooth Gouraud gradients, no toon steps in any of the 16,937 images (all
  reviewers, E, H). Hair seen from behind in a 4x emulator render: brightest #73514A facing the camera to #312031 at the
  silhouette, ratio (0.43, 0.40, 0.66), violet shift, no bands (S#2322) [R4, M]. Section 2 CONFIRMED.
* Whites and fur shade lilac IN GAME in CF: K.K. fur lit 10% #F9F8F9 to dark 10% #666292 (0.41, 0.40, 0.58) and #6E6690
  (0.44, 0.42, 0.58) (S#8165, 8166) [V1]. NL catalog renders agree in direction: snowman (0.72, 0.70, 0.85), fridge (0.56,
  0.56, 0.64), white furniture (0.62, 0.59, 0.67) [R5, Rn, M]. Section 2's "CF whites neutral grey" (render) is replaced.
* Foliage: leaf cards lime at the tip #79BF46 to #8FD648 (hue 91 to 106), dark teal at the base #04402E to #053422 (hue
  160 to 168) (CF S#3, 46, 277 Nookipedia, plus one Fandom WEBP copy) [R1, M]; NL canopy p5 #00483F, median #07865B, p95 #3DE04A (S#12401) [R6, H for
  direction].
* Specular: none on terrain or matte cloth; fruit carries a small hard white dot (CF S#3; icon S#4095 #FFFFFF disc) [R1,
  R3]; NL glossy plastic, lacquer, chrome and upholstery show broad soft sheen bands, glass and ice white edge glints
  (catalog renders) [R4, R5, M]; CF gold furniture shows horizontal bright and dark bands that ignore relief, likely an
  environment map (S#8008, by eye; its colours came from a WEBP copy) [R1, L].
* **Grades (multipliers on the day value of the same surface, displayed; linear by the 2.2 power [S]):**

| grade | multiplier | evidence |
|---|---|---|
| NL night, mid tones | (0.34, 0.31, 0.70), linear (0.09, 0.08, 0.46) [audit, mean of two methods, item 29]; by surface R 0.28 (path) to 0.45 (flowers), B 0.68 to 0.77; V3's (0.45, 0.44, 0.75) and flower beds (0.44 to 0.50, 0.42 to 0.49, 0.76 to 0.81) are the light end | [V3, audit] day S#12062 against night S#12036 (correlation 0.70 to 0.71); M. R6's (0.46, 0.50, 0.53) REJECTED |
| NL night, whites | not darker than mid tones: p90 ratios 0.44 to 0.57 in R and G; V3's (0.26, 0.25, 0.65) not reproduced [audit] | same pair; L |
| NL night skin | about (0.2, 0.3, 0.75) | [R3] catch poses, rough masks; L |
| NL dusk | R 0.70 to 0.96, G 0.48 to 0.69, B 0.59 to 0.75 by surface (flowers and fence (0.92 to 0.96, 0.65 to 0.69, 0.66 to 0.75), red roof (0.70, 0.48, 0.59)); midpoint (0.83, 0.58, 0.67) [S] | [V3] S#11985 against 12000, 12005, 12015; a magenta tint, not one uniform multiply |
| NL dusk skin | #E7826B from day #FFD3A5 | [R3] L |
| CF night (ground) | (0.64, 0.68, 0.83) on dirt; grass G 0.59, B 0.87 | [V1] S#1095, 149 against S#1129, 8290, 8495 |
| CF night (characters) | (0.35, 0.37, 0.82) | section 2 [A], not re measured |
| CF dusk (ground) | (0.87, 0.66, 0.65) | [V1] S#253 only; L |
| CF rain (ground) | (0.57 to 0.58, 0.53 to 0.58, 0.79 to 0.90) | [V1] S#158 |
| NL rain | desaturates and darkens without fog; salmon cliff to #A78DA6, skin #CEAEA5 | [R4, R3] M to L |

* **NL interior light model** (H for the measured rooms):
  * Ambient is violet, never black: lights off reads (80, 70, 166) #5046A6 on the right wall, #5A4DAE on the floor
    (S#12260) [V3, R6].
  * Lamps ADD per channel in display values, clamped at 255: Level 1 wall increments red (+98, 0, 0), indigo (+99, 0,
    +87), orange (+97, +47, 0), yellow (+97, +96, 0), light blue (0, +47, +87); untouched channels stay within 2 levels
    (not a multiply); floor about 0.84 of the wall. Level 2 doubles Level 1, Level 3 adds half a step, Level 4 equals
    Level 3: ambient + colour x min(level, 2.5) x 98 [V3, 21 frames]. Red reads magenta (#FE44A5), yellow cream. In linear
    light +98 on this wall moves R from 0.080 to 0.445 by the sRGB curve (0.078 to 0.453 by the 2.2 power), so do the add
    in display space or tune per surface [S, audit].
  * House rooms: NEUTRAL key (back wall whites median #F7F7F7, a quarter clipped) and lavender falloff on surfaces facing
    away: side wall over back wall (0.68, 0.70, 0.82), front floor over mid floor (0.67, 0.73, 0.77), about 290 rooms; on
    S#2204 the floor runs #EFE7FF to #C6C7E7 (0.86, row 210) to #BDBEDE (0.82, row 225) [V2, LL]. R4's lilac key REVISED.
  * Ramp to black toward the ceiling over about 110 px (S#10458 column x 380: 1 to row 45, then 33, 76, 119, 167, 194,
    231, 251, 254 every 15 rows) [R5, V2]; showroom rings 1.01 to 1.11 out to r 0.8, then 0.71, 0.45, 0.17, 0.05 at r 0.9
    to 1.2 [R3, M]; pure #000000 beyond the walls [R5, V2]. Interiors CLIP: 4.4% of S#2204 is #FFFFFF, 26% of S#10457 and
    10458 at 250 or more [V2]. The room light tints the player too [R5]. Window light is a pale floor square (E) [R6].
* CF interiors: museum, K.K. stage and Katrina's tent are low key with one warm key and black surroundings; museum wall
  #251B11, floor #4F3A28 (light #8F6C46) (S#958) [R1, M].
* **Emissive windows, lamps, glow:** CF windows are a constant #FFFFA0 day and night (S#382, 1095), town hall #F8FB9D
  [V1, H]; CF lamp glass clips white to about 45 px, ring #9B947F to 65 px, background by 90 to 130 px in an 896 px frame
  (S#758) [R1, M]; handheld fireworks light a warm ground pool (S#8757, WEBP copy, by eye only); CF flames end in pink (campfire p50 #FF888E,
  p90 #FFD436) [V1, Rn], pale core #FFDE8E (S#4854) [R2, Rn]. NL night windows #F1C791 to #FCD45E (S#2543, 1589, 9215, 10226, 10672) [V2], clipping to #FFE0D1
  (S#13760) [R6]; day windows clip #FFFFFF [R4]; warm pixels 0.3 to 5% of a night frame [R3, R4]. NL glow at 400x240 [R3,
  M]: warm lamp core #FFFFA2, #FFFE10 at r 6, #F2AC0A at r 12, #6D4410 at r 20 (S#16603); white street lamp #FFFFF8, gone by
  r 12 (S#16417). Additive headlight wedge #A19568 (S#1589) [R4]; campfire ground light #833F1C [R5]; fireworks and aurora
  additive (E).

### 16.6 Shadows

No cast shadow from a character, tree or building in any retail frame; every shadow is a decal centred under its object
with no sun offset, and every NL shadow multiplies blue least (all reviewers; [R4] on floor, sand and snow; H).

| shadow | displayed ratio to the ground beside it (R, G, B) | evidence |
|---|---|---|
| NL player, outdoor grass | core 0.41 to 0.50 CONFIRMED on 40 more emote crops: (0.46, 0.44, blue 2.56x on lime) | [R3] H |
| NL player, open sand | (0.55, 0.55, 0.67), luminance 0.56 | [V2] S#1539 LL; M (one frame). R5's S#9175 value REJECTED: the player stood in a palm shadow |
| NL player, gameplay grass | 0.40 to 0.65 depending on the reference patch | [V1] S#16441; L |
| NL player, interior | (0.62, 0.58, 0.67), luminance 0.61, against the floor on the same rows; V2's (0.52, 0.52, 0.59) divided by a brighter floor further back | [audit, V2] S#2204 LL; M |
| NL overlap | palm shadow (0.68, 0.66, 0.78) plus blob gives luminance about 0.45, lighter than the product 0.38: stacked shadows only partly multiply | [V2] S#9175; M |
| NL palm canopy on sand | (0.58, 0.56, 0.68), star shaped frond outline, centred | [R4] S#1539; M |
| NL tree canopy on snow | (0.60, 0.60, 0.67); about 0.63 under a snowy tree | [R4] S#1242; [R3] S#3087; M |
| NL floating balloon on snow | two level ellipse 49 x 28 px (aspect 1.75): ring (0.57 to 0.65, 0.58 to 0.64, 0.68 to 0.75), core about (0.36, 0.38, 0.41) [S from R5 pixels] | [R5] S#9489; M |
| NL night | blobs barely visible | [R4] S#2005; L |
| CF player on snow | core 0.55 to 0.64 (dark 10% 0.54), tint (0.53, 0.53, 0.58), width at the 0.85 level about 54 px (about 0.5 P), 8 to 10 px ramps | [V1] S#111; M |
| CF emote crops on grey floors | no shadow (0.98 to 1.02): edits, unusable | [R1] |
| furniture, trees, holes | soft dark contact blobs; dug holes and pitfalls have dark rims | [R4, R6] E |

Baked contact shadows in CF floor textures (S#5740, 5778) are darker, cooler versions of the ground: supports multiplying
the ground rather than pasting a dark sticker (section 15) [R2, E].

### 16.7 Models and shapes (renders, catalog and in game crops; E unless a number is given)

* Mesh reuse is the rule: furniture series swap textures on one mesh (15 dresser colours S#15686 to 15700), gyroids share
  a texture across three sizes, one RV van mesh, one house room box, one 8 sided shirt mesh on a wooden mannequin [R2 to R5].
* Silhouettes, Douglas Peucker corners at 1% of the diagonal (coordinator metric): furniture median 20 (IQR 15 to 28, n
  802), headgear 21, gyroids 29, fish models 31, bug models 46.5, CF character art 40 [R2]. Lathe props 8 to 12 segments,
  barrels about 12 staves, boulders 20 to 30 faces, pond rocks 10 to 14 flat facets with violet darks p50 #53535B [R2].
* Buildings: squat, roof as big as the walls, chunky trim (CF house roof #A53B29, walls #A84637, trim #EDEAD6; museum roof
  #6678AC; renders) [R2]; CF scallop tiles with lighter lower edges, half timber, arched doors with a glowing round window
  [R1]; NL house (396) and roof (34) crops show bevelled slabs with bulging eaves, each tile row painted with a lighter top edge and a
  darker lip (red p5 #912B39, median #BB3F5B, p95 #D5616C), chimneys with one or two white smoke ellipses [R4, R6].
* Trees: short fat trunk with a 4 to 5 lobe star root flare; bark #905143 to #B86A6A with cream ring knots #DE9F71 to
  #E8D39E and crevices #4D222B [R1, R2]; canopies of drooping leaf cards with lighter tips (oak #005B49 / #079048 /
  #6BB52C) [R2]; NL five lobed cards about 20 px wide in a 170 px crop [R6]; blossom #F699D1 [R5], maple #963525 [R3];
  palms with chevron trunks; snow caps as separate puff balls.
* Props: fruit large glossy spheres with one white dot [R6]; bridges (stone arch, steel truss, planks with rope posts) span
  the full river width, deck about 0.35 of the plinth [R5, R6]; fences chunky L segments [R5]; CF fish models 1.83 to 2.17
  long over high, honeycomb scale cells, countershading with a zigzag boundary, binary alpha fins [R2, H].
* Characters: face decals, flat orange red nose, cheek blush, skin disc ears, chunky hair shell with a zigzag fringe [R2,
  R4]; skin CF in game lit #F5B88F to #F7B991, shade #D6947E to #D99582 (S#1118 to 1121) [R1], CF art #FEBA90 to #FFC296
  [V1], NL day #FFD3A5 [R3]. Umbrellas: faceted domes of 6 to 8 panels [R1, R3].

### 16.8 Textures, formats and the grain target

| asset (CF unless NL) | finding | evidence |
|---|---|---|
| flooring textures | 94 of 94 are 256x256 and exactly mirror symmetric in both axes: a 128x128 painted quadrant shown mirrored; colours median 53 (18 at 16 or fewer, 1 at 256), 88 of 94 on the ROUNDED RGB565 grid (round(q x 255/31), only 4 on the bit replication grid of the lossless frames [audit]); grain median 3.6; light baked from the top left (bevels, drop shadows) | [R2, V1] T; H |
| wallpaper textures | 95 at 128x128 indexed, 64 to 256 colours (median 139), 83 to 94% of R values on the 5 bit grid, painted light and shade baked in, grain 4.13 | [R3] T; M |
| ground swatches | 133x133, four 64x64 quadrants per family (16.3) | [R3, V1] T; H |
| stationery | 68 sheets 512x384, median 100 colours | [R3] T |
| clothing | the 346 "Texture" files confirm section 6; exclude the 346 "(CF).png" re encodes; one stochastic dither exception (Snowcone S#6965, 2x2 checker share 0.0142); gradients banded in 1 to 2 texel steps | [R2, V1] T; H |
| Wendell and preset designs | 32x32 opaque; Wendell median 7.5 colours, NEAR RGB555 (17 of 52 strictly on the grid, median file 95% of values); presets median 4 colours | [V1] T |
| bug and fish icons | 64x64, binary alpha, at most 15 colours; darkest ring over next ring inward 0.71 to 0.80 (section 6 metric CONFIRMED on 128 more icons); ring over body 0.56 to 0.59; fish head left | [V1, R2] T; H |
| emotion icons | 80x40, alpha on the 3 bit grid (union of all 8 levels, 4 to 7 per icon, median 6), 12 to 15 colours, props outlined in WHITE 1 px | [V1, R2] T; H |
| map icons, inventory icons | 15 to 40 px with 16 colours; 132 icons at 64x64 with 15 colours plus 1 bit alpha | [R3] T |
| render filtering | CF item renders nearest filtered (about 36 texel steps across a 684 px apple), shirt renders bilinear: viewer filtering differs per category | [R3] Rn; M |
| NL shirt texels | about 1 native pixel at gameplay distance | [R6] S#10976 crop of a 3.5x upscale; L |

Compression facts (H unless marked): Fandom serves lossy WEBP from its CDN (all 8,816 corpus copies; 0 of 1,773 screenshot
copies pass RGB565); appending `&format=original` returns the upload [R5]. On the 13 lossless Fandom frames the WEBP copy
differs by a mean 2.3 to 4.6 levels, p99 per pixel maximum error 26 to 77 [V2]: never take colours from WEBP copies. The
WEBP shift on grain medians is small (2.91 against 2.89 on 453 frames [V3]; 1.97 against 2.09 on the 13 lossless [V2]).
A JPEG at any quality raises grain on outdoor 565 frames (quality 80 4:2:2 adds median +0.14, IQR +0.05 to +0.25;
quality 95 4:4:4 still +0.13) and lowers it on flat UI frames (encyclopedia 1.73 to 1.38) [V3]. CF 832x456 frames are 640
px wide resampled (period 4.33 px). GIF uploads are palette limited [R6].

**Grain target (|L minus 3x3 box mean| median, Rec 601 L) [V3 reconciliation].** Retire 2.95 (section 15): it pooled 21
mixed press and lossy captures. The spread between reviewers was CONTENT: lossless medians run 1.73 (encyclopedia UI),
2.08 (26 sky frames without balloon), 2.34 (284 scenes), 2.56 (75 outdoor frames with grass over 0.2, IQR 2.19 to 3.03),
2.82 (48 RV rooms), 3.83 (313 house rooms with patterned wallpaper). A clean render lacks the 565 steps, which add 0.15 to
0.26 (about 0.20) on our renders. NEW TARGET for a clean outdoor render: **2.35** (accept 2.0 to 2.8), measured exactly as
`grain_compare.py` does (BILINEAR downscale to 400x240, no 565 snap, no JPEG). Alternative: snap the render to RGB565 in
QA and compare with 2.56 directly. Cross checks land at 2.2 to 2.4 (album JPEG outdoor without balloon 2.66 minus 0.14
minus 0.20 = 2.32). The resize filter alone moves a render's grain by 0.25 to 0.29 (BILINEAR against LANCZOS), so state the
method with every number. CF retail 832x456 PNG grain 1.27 (IQR 0.61 to 1.82), 420x308 crops 1.71 [V1]: CF scenes are
cleaner. The test town (1.53 to 1.76) is still 0.6 to 0.8 below the NL target.

### 16.9 UI and design (for original on screen callouts; never copy the game's assets, section 8)

* NL dialogue balloon: bottom 40% of the top screen (rows about 160 to 230), fill a dark olive black, median #1F2018 (p10
  #070F08, p90 #313231, 400 frames) [V3, REVISED from navy and teal]; R3 saw its median follow the background (#102439 over
  sea, #000818 at night), so it is translucent (M); fine dot texture, wavy or scalloped top edge, a cream and white checker
  rim (pink for announcements) [V3, R4, E]; white text with coloured keywords, orange arrow cue [R4]; a coloured name pill
  per speaker (Tom Nook #34CF33 [R4]); the "pink female, blue male" rule is contradicted (S#10501) [V3].
* NL HUD: cream date pill #F8E396 with time #D99647 bottom left [R4], bells pill top right, green tour timer pill top left.
  Catalog: gingham with a 32 px period in #FCCA7D, #F7DA8B, #F9DFA7 (S#15562, JPEG original) [R5, H], bell pill #E2C35E to
  #E5C01D, pink name bar #F44684, item small and centred on a vast empty field. Encyclopedia: header fish #0763CC, insect
  #0C8D08, background #C7D8FE with a darker wave print, footer #ECDFA9 to #F6E3A1 [R3]. Badges: gold oval with three
  medallions; QR screens on pink gingham.
* CF: dialogue cream fill about #FFEFA6, translucent (alpha about 0.73, L), opaque scalloped yellow border, coloured
  speaker pill, coloured key words [R1]; gold ring clock with a date pill top left; encyclopedia frame #5C8AC8, panel
  #D2EDE9, title pill #A3E2CB, slots #A3B657, arrows #824AD2, progress #F1CB54 [R1, M]; map tiers #4A9621 and #5AAA21,
  rivers #085DCE, frame #9BE0B4 [R1, M].
* Composition (E): flower dense ground, round canopies framing the top, one saturated accent near the player (red
  mailbox, fruit, bobber); dusk floods the frame with magenta and violet; night keeps navy with one or two warm windows as
  the focal points; houses framed centrally with the sky filling the top 40 to 65% [R4, R5, R6].
* Callout guidance [S]: a dark translucent rounded panel in the lower 40% (fill near #1F2018; its alpha is a design choice, the game value is unmeasured) with a
  light scalloped rim of your own design, white text, one coloured keyword, and a pill shaped speaker tag in a nature
  colour reads as the same family without reusing any asset.

### 16.10 Shader layer stack

Layer order, bottom to top (every material uses a subset, in this order): **B** base painted texture; **M** motif print or
painted detail; **L** LUT light (section 2 and 12.2: smooth ramp on N.L from the camera side headlight, shadow end violet);
**R** rim (NL characters only); **D** decal shadow multiply (separate Blended decal objects, linear values); **H** haze (NL
only, mix by view depth; ground and water materials only, never objects, 16.17); **G** time of day or weather grade (multiply on the lit result); **E** emission; **T**
transparency; **P** post (glow, clamp, optional 565 snap). Values are displayed sRGB unless marked; NL retail first, CF
after a slash.

| material | ordered layers with values | sections |
|---|---|---|
| grass ground | B NL near #0D7541 (spring and summer; late winter #5B8D3B, December #7B673A, island #187939, Main Street #5A9E5A) / CF #0A8C54; M NL dots, stars, leaves or diamonds per season, darker print p10 #005A3D / CF one family per town (squares, triangles, circles) in 3 motif colours (#60AF17, #02AE58, #0AA77B); L; D player blob 0.41 to 0.50 (linear 0.14 to 0.22), blue lifted; H mix toward #88BF68 to #B5E36B over the 100 rows below the horizon, far over near luminance about 2.15, saturation to about 0.5; G night (0.34, 0.31, 0.70) / CF (0.64, 0.68, 0.83), dusk (0.83, 0.58, 0.67) / CF (0.87, 0.66, 0.65) | 16.3, 16.5, 3 |
| path, dirt, plaza | B CF #C79748 with blobs #BD944A and specks #D68C31 / NL bank #946D31, plaza dirt #9D6223, cobble #AAA9B9; M CF hexagon pebbles, NL round cobbles; L; edge: CF dark brown lip on the dirt side #7E5018 (no teal outline), NL soft blend; D; H NL cobbles toward a lighter lavender grey (#ADB2C6 near, #C6C3D6 far, luminance x1.11, S#3152), not toward the grass lime; G as grass, CF rain (0.57, 0.55, 0.85) | 16.3 |
| sand and wet sand | B NL #EFB67B to #F4BB8C (town #CA9A68 to #DDA374) / CF shore #E7CB8B, inland #CEA64A; M shell decals, CF orange hexagon pebbles; wet band = B x 0.62 to 0.71 luminance, redder; foam line #ECF2F8; L; D player (0.55, 0.55, 0.67), palm (0.58, 0.56, 0.68); H; G dusk wet sand toward #86779D | 16.3, 16.6 |
| snow | B NL lilac #A1A6E0 (range #8E90BA to #E7D7DA, warm highlights #FFEFEB) / CF #BDD3E6; M NL stars or hex cobble, CF the town family swapped (rounded squares, stars, hexagons) #F0F6F4; L shade toward #A1BDC6 / violet; D canopy (0.60, 0.60, 0.67), CF blob (0.53, 0.53, 0.58); G night to #4E56B1 to #5989EE, 5:33 PM pink #C499BC | 16.3, 16.6 |
| cliff | B NL salmon #FCC599 lit, #B79EB1 mauve slabs / CF wall #86645D; M horizontal strata, CF cream band #D9C399 and crevices #4E1F36; L; grass lip on top; G rain toward #A78DA6, CF winter swap #5E5E79 | 16.3 |
| water surface | B NL sea #3A76C4 (troughs #1770C2, crests #8FADD1), river #316BAF / CF sea #1C5196 to #183B8B, river #2886C5 (section 4.3); M cellular lines +21 to 42% at p95 (CF tile per section 4.3), crest scallops; L none; H far band #95B3F2; G dusk toward mauve #70455F, night #0E1D53; T NL translucent near shore (#A5C3E7 to #C6D3E7 over sand, section 15 alphas 0.45 to 0.70), CF opaque | 16.4 |
| underwater | darken submerged surfaces by depth toward #3B6DAC (section 15) | 4.3, 15 |
| fish shadow | separate decal just under the surface, drawn after the water; crisp 1 to 2 px edge (no blur), plateau multiply (0.50, 0.63, 0.68) displayed = (0.22, 0.36, 0.43) linear; length 0.16 to 0.82 P by class, width 6.5 to 11.3 px at 240 rows / CF soft teardrop 3:1, 0.46 to 0.73 | 16.4 |
| tree canopy | B leaf cards (binary alpha clip) lime tip #8FD648 to teal base #04402E (NL p5 #00483F, p50 #07865B, p95 #3DE04A); seasonal swaps (blossom #F699D1, maple #963525); L; snow puffs as a separate shell; D ground canopy shadow (0.58 to 0.60, 0.56 to 0.60, 0.67 to 0.68) | 16.5, 16.6, 16.7 |
| trunk | B bark #905143 to #B86A6A; M ring knots #DE9F71 to #E8D39E, crevices #4D222B; star root flare geometry; L; D contact patch | 16.7 |
| flowers, fruit | B saturated petals and fruit; M painted detail; L; fruit: one small hard white dot (painted, not a shader specular); D clump contact 0.68 (section 15) | 16.5 |
| rock | B slate mottle p50 #53535B with violet darks; faceted mesh 10 to 30 faces, flat shaded facets; L violet shadow end; D contact band | 16.7, 2 |
| building wall | B painted planks, plaster or half timber with baked bevel light; L; G grade; E windows NL night #F1C791 to #FCD45E (clip #FFE0D1), CF #FFFFA0 always | 16.5, 16.7 |
| roof | B tiles with painted lighter top edge and darker lip per row (red median #BB3F5B); L; snow cap shell in winter; G | 16.7 |
| interior floor | B 256 mirrored painted texture with baked top left light; lighting = violet ambient (80, 70, 166) PLUS lamp colour x min(level, 2.5) x 98 in display values, clamp at 255 (whites may clip); neutral key toward the back, front edge 0.82 to 0.86; D blob about (0.62, 0.58, 0.67) [audit, same row reference]; P vignette to #000000 outside the room box | 16.5 |
| interior wall | B 128 indexed wallpaper with baked light; same light model; side walls (0.68, 0.70, 0.82) of the back wall; vertical ramp to black over about 110 px toward the ceiling | 16.5 |
| furniture | B series hue texture with baked gloss streaks (gold #8D6000 to #FFD374 ramp, chrome streaks); L; glossy plastic: broad soft sheen band, glass and ice white edge glints, no sharp specular; D soft contact blob | 16.5, 16.7 |
| character skin | B NL #FFD3A5 / CF #F7B991; face decals; L shade toward #D99582 (CF in game) or #CE8F79 (NL, section 7.3), lilac at the silhouette; R NL rim (section 2); D blob; G dusk #E7826B, night about (0.2, 0.3, 0.75) | 16.5, 16.7 |
| hair | B low contrast mottle shell; L hair edge ratio (0.43, 0.40, 0.66) from the brightest to the silhouette; R NL head top rim | 16.5, 2 |
| clothing, white fur | B CF 32x128 strip, 16 colours, banded gradients; L whites to lilac (0.41, 0.40, 0.58) | 16.5, 16.8 |
| sky | B view elevation dome: NL zenith #2120E7, walking strip #2040F0 to #2050ED at the top of the frame to #2166FA to #2172FD at the ground, deep plazas lilac #5A65F7 last row; CF #1714C9 body, ramp over the last 19 to 28% of a look up frame to #1A5CDC to #1B6CE2; night NL flat #01215A (violet variant #424169), CF #000B39 / early night #312848 to #413357; E stars single pixels, moon disc #F0F3C6 with halo | 16.2 |
| clouds | unlit cards with soft alpha edges beyond the bent ground; NL #CCE1FE (p10 #A6C5FC, p90 #DDEAFD) or flat puffs #949EF7 with tops #FFF3CE; CF cores #C1DCFF, edges #B1CFFF, 15 to 30% cover in look up views; dusk lit clouds #DDA495 | 16.2 |
| emissive lamps, fire | E lamp core #FFFFA2 to #FFFFF8; P glow: warm halo to about r 20 px (15 to 20 px) or white tight to r 12 at 400x240; fire as blob cards #FFDE8E core to pink rim #FF888E; additive beam wedges #A19568; warm ground pools under handheld lights (by eye) | 16.5 |
| UI | flat, unlit, no grade; dark translucent balloon #1F2018, cream pills, gingham 32 px; Closest filtering for pixel art | 16.9 |

### 16.11 Corrections to earlier sections (details and evidence in the subsection named)

| section | old value | new value | where |
|---|---|---|---|
| 0, 13 | NL retail rests on NL#938 to 945; no environment textures | 737 lossless frames, about 3,440 album JPEGs; CF ground, floor and wallpaper textures | 16.0, 16.8 |
| 1.2, 12.1 NL horizon row | 0.165 | walking 0.144 (H); talk and catch 0.18 to 0.19; 0.165 press only | 16.1 |
| 1.3, 12.1 NL bend | 13 P | about 14.5 to 15.7 P for the 0.144 row (L) | 16.1 |
| 1.2 NL interior | player 38% (press) | 30.8%, feet 78.8% (313 lossless rooms) | 16.1 |
| 1.2 CF interior feet row | 0.636 | 0.62 (M); share 23.0 to 25.4% by hairstyle, 25.2% inside it | 16.1 |
| 4.1 CF sky | #1A2AD0, 13% band | retail #1714C9 to #1210BE; look up ramp 19 to 28% to #1A5CDC to #1B6CE2; #1A2AD0 is press | 16.2 |
| 4.1, 12.1 NL sky | #2135E7 to #4078DA | zenith #2120E7 (LL) / #2121E5 (JPEG); walking strip #2040F0 to #2050ED (top) to #2166FA to #2172FD (ground); variants #293CE7, #2149F7 | 16.2 |
| 4.1, 15 clouds | white puffs key art only | CF look ups 15 to 30% (#C1DCFF); NL cumulus #CCE1FE and puffs #949EF7 in retail, rare | 16.2 |
| 4.1, 4.4 night | NL "blue violet", sky always lightens to the horizon | NL #01215A flat; violet family a minority; CF #000B39 and early night violet bands | 16.2 |
| 2, 15, 12.1 haze | NL retail unproven or likely | H: far over near 2.15, upper 100 ground rows, toward #88BF68 to #B5E36B | 16.3 |
| 12.2 step 8, 16.16 haze | retail off; then haze on every object | on in NL retail, ground and water only; objects at the horizon keep their colour; cobbles toward lavender grey | 16.17 |
| 4.2, 15, 12.1 NL grass | #92DB10, then #568F4F to #5D9D49 | near #026637 to #107442 (spring, summer); seasonal set; lime emote only | 16.3 |
| 4.2 CF motif | shapes vary per frame | one family per town (squares, triangles, circles), four seasonal palettes | 16.3 |
| 4.2, 10, 12.1 CF ground outline | teal (0,41,62) 2 to 3 px | brown lip on the dirt side #7E5018 (teal was press) | 16.3 |
| 4.2, 12.1 NL sand | #FED08A (press) | #EFB67B to #F4BB8C; town beach #CA9A68 to #DDA374; wet 0.62 to 0.71 | 16.3 |
| 4.2 CF cliff | #CEA88A | #86645D, band #D9C399, crevices #4E1F36 | 16.3 |
| 4.2 snow | key art only | NL lilac #A1A6E0, never white; CF #BDD3E6 | 16.3 |
| 4.3 NL caustic period | 115 to 120 px | cannot confirm in retail; translucency CONFIRMED | 16.4 |
| 15 fish silhouette | soft teardrop, #0E2448 at 0.6 | crisp 1 to 2 px edge, multiply (0.50, 0.63, 0.68), 0.16 to 0.82 P | 16.4 |
| 2 night, dusk | NL not measured | NL night (0.34, 0.31, 0.70), dusk (0.83, 0.58, 0.67) midpoint; CF ground night (0.64, 0.68, 0.83) | 16.5 |
| 2, 15 interiors | not covered | neutral key plus violet ambient falloff; additive lamps on (80, 70, 166) | 16.5 |
| 2 exposure, 9 item 9 | never clip, no pure black | interiors clip and are #000000 beyond the walls; outdoors unchanged | 16.5 |
| 2 CF whites | neutral grey (render) | lilac in game (0.41, 0.40, 0.58) | 16.5 |
| 3 blobs | NL 0.41 to 0.50; CF 0.28 to 0.60 press | NL CONFIRMED, sand 0.55, indoor about 0.61, partial stacking; CF snow 0.55 to 0.64 | 16.6 |
| 6 textures | no environment textures | floors, wallpapers, ground swatches, Wendell, emotion icons; drop "(CF).png" copies | 16.8 |
| 7.4 NL dialogue | #0F0B07 to #241810 | #1F2018 olive black, translucent | 16.9 |
| 15, 12.2 grain | 2.95 | RETIRED: 2.35 clean (BILINEAR to 400x240), or 2.56 after a 565 snap | 16.8 |

### 16.12 Open questions

1. What decides the light walking sky strip (#2044F2 top) against the deep minority (#2121E5)? Needs a frame series of
   one town at one hour with known places. (The lossless #005DC6 to #005DCE "mode" is resolved: encyclopedia UI, not sky.)
2. Retail camera solve: no square grid outdoors in a retail frame yet; pitch and lens still come from press NL#950. The
   bend radius for the 0.144 horizon (14.5 to 15.7 P) inherits that.
3. NL night zenith: when is it navy #01215A and when violet (#424169 family)? Hour, weather or season cannot be confirmed.
4. Dusk: the families in 16.2 are one frame each; a dated series would turn them into a ramp by hour.
5. Fish shadow plan shape (teardrop or oval) at retail resolution, and whether NL caustic cells really shrink in retail.
6. NL interior light under lamps in ordinary house rooms (only the tint series room was solved) and CF interiors at all.
7. CF in game rim, CF haze on open retail ground, CF night character multiplier: still not re measured.
8. Grain: libjpeg is not the 3DS encoder; the lossless outdoor sample (75) includes island and museum frames.
9. Name pill colour rule and the dialogue balloon's exact alpha.

### 16.13 Audit log

Final audit of this section: 38 decision relevant numbers re measured with independent scripts in
`verify_shots/AUDIT/` (helper `au.py` loads the Fandom original upload where one exists, never a WEBP copy), on a third
sample or with a third method where possible; arithmetic rechecked; writing rules checked. Status CONFIRMED means the
audit value agrees within the stated range or within a few 8 bit levels; CORRECTED gives old, new, method and IDs.

| # | number | status | audit method and result | IDs |
|---|---|---|---|---|
| 1 | NL walking horizon, ground starts at row 34 or 35 of 240 (14.2 to 14.6%) | CONFIRMED | `a01_horizon.py`: per column first non sky row (sky: B over 150, B over G+25, B over R+60); median 34 or 35 on all 8 listed frames. Third sample `a01b_scan.py` over every 400x240 item (modal transition row with green below, at least 40 agreeing columns): 11 more walking frames at 34 or 35, one at 33 (S#10317) | S#1351, 10394, 11519, 2657, 2697, 10649, 10678, 13257; new S#1341, 2694, 9207, 9409, 10120, 10317, 10507, 10638, 10932, 11143, 13617 |
| 2 | NL talk and catch horizon rows 43 to 46, a few at 39 | CONFIRMED, range widened | `a01b_scan.py` third sample: 26 frames with a sky to grass transition in rows 39 to 47, mode 43 to 44 (14 frames), two at 47 (S#1250, 10115), two at 40, one at 39. Body now reads "a few at 39 to 40 or 47" | S#3436, 3468, 3506, 10779, 11162, 13224 and 20 more |
| 3 | NL walking player height 73 to 76 px, about 30% | CORRECTED | old 73 to 76 px (about 30%); new 76 to 77 px, 31.7 to 32.1%. Method: 6x NEAREST crop with a row ruler plus pixel dump: hair top at row 61 to 62 (first lilac pixel), sole bottom at row 137 to 138 (last shoe pixel above the shadow) on the Q80 original. Feet row 57.5% lies inside the stated 44 to 63% | S#12899 |
| 4 | Fish shadow lengths 10.4, 22.8, 24.0, 38.0, 44.0 px; widths 6.5 to 11.3 px | CONFIRMED (widths within 1 px) | `a23_fish.py`: water reference from the box median outside a dilated mask, flood fill at lum ratio below 0.85, PCA extents: lengths identical; widths 6.5, 7.3, 10.1, 12.1, 12.0 (large and very large about 1 px wider than 11.2 and 11.3 because the extent counts the end pixel) | S#12898 to 12902 |
| 5 | Fish shadow multiply (0.50, 0.63, 0.68) displayed, linear (0.22, 0.36, 0.43) | CONFIRMED | same script, per channel median ratio inside the mask: R 0.45 to 0.60, G 0.61 to 0.67, B 0.65 to 0.73; median over the five classes (0.48, 0.64, 0.67). Arithmetic: 0.50^2.2 = 0.218, 0.63^2.2 = 0.362, 0.68^2.2 = 0.428 | S#12898 to 12902 |
| 6 | Fish shadow edge: plateau 0.55 to 0.63, boundary 0.70 to 0.75, one pixel outside 0.91 to 0.95 | CONFIRMED | `a23b` (inline script, third method: erosion depth rings instead of a profile): depth 2 to 3 inside 0.59 to 0.65, edge pixel 0.70 to 0.75, first outside ring 0.90 to 1.01, second 0.93 to 1.04 | S#12898, 12900, 12902 |
| 7 | NL house interior player 30.8% (hair row 116 to sole row 189), feet 78.8% | CONFIRMED | own per pixel median of the same 313 rooms (all 313 pass the exact RGB565 expansion test) shows the same player; single frame column dumps (rooms 1, 101, 201, 301 of the list) put the hair top at row 115 to 116 and the last pink shoe row at 189: 74 to 75 px, 30.8 to 31.3%; 189/240 = 78.75% | `V2/house_ids.npy` list, S#1608 and others |
| 8 | CF interior player 25.0 to 25.4%, soles 61.6 to 61.8% | CORRECTED (share) | old 25.0 to 25.4%; new 23.0 to 25.4% depending on the hairstyle. Pixel dumps: S#433 hair top row 178, sole row 282, 105/456 = 23.0%; S#541 hair top row 166, sole row 281, 116/456 = 25.4%. Soles 282/456 = 61.8% and 281/456 = 61.6% CONFIRMED | S#433, 541 |
| 9 | CF look up horizon 72.3 to 73.5% where unobstructed; 76.2% on S#8495 | CONFIRMED | `a09b_cfmode.py`: modal last sky row among columns: 72.5 to 74.0% on all 8 listed frames (S#8202, 8290, 8303 at their second mode 348 to 350, the first mode being a roof); a one row offset from V1 (last sky row against first ground row) explains 72.3 against 72.5. S#8495: modal 73.3% at the left bank, the sand rises to about 76% mid frame (by eye on a marked copy), so 76.2% is a terrain level, not a camera change | S#7710, 8279, 622, 112, 8202, 8290, 8303, 8353, 8495 |
| 10 | CF day sky body #1714C9 (house look ups), #1210BE to #1211BF (640x480 look ups), top #0E0EB9 to #100EBA | CONFIRMED; label and count CORRECTED | `a10_cfsky.py` (mask B over 150, B over G+25, B over R+60, min channel under 120; body rows 15 to 45%): house look ups #1715CA (p10 #1714C9, p90 #1816CB, 73 frames), top #1511C5; 640x480 set #1514C2 (3 to 4 levels lighter because my band reaches lower), top #100DB9. Corrections: the house look up set S#316 to 411 is 75 crops at 420x308 (R1 inventory, sizes in `corpus_items.json`), not 832x456; the 640x480 list names 12 frames, not 13 | S#316 to 411; S#112 to 114, 8036, 8202, 8290, 8303, 8321, 8353, 8378, 8407, 8787 |
| 11 | CF look up ramp 19.4 to 27.9% of the frame, median 24.3% | CONFIRMED | `a11_cframp.py` (third definition: rows whose median sky G stays above the body G + 8 down to the horizon): 21.5 to 25.4%, median 23.6%, 14 frames; horizons 340 to 354 of 480 | same 12 plus S#7710, 8279 |
| 12 | CF cloud cover in the top 20%: median 29% with soft edges, 18% cores; use 15 to 30% | CONFIRMED | `a12_cfcloud.py` (third threshold: R over clear sky R + 40 with B over 150; cores min channel over 190): soft edges median 35% (p10 22, p90 44), cores 11% (p10 7, p90 19), 75 frames. The recommended 15 to 30% band sits between core and soft cover and stands | S#316 to 411 (420x308), 640x480 set |
| 13 | Lossless retail frames: 737 (721 Nookipedia PNG, 16 Fandom original PNG) | CONFIRMED, wording CORRECTED | `a29_565.py` (strict bit replication on every pixel, actual size 400x240, original upload preferred): 719 (704 Nookipedia, 15 Fandom: S#9165 to 9177, 12702, 13638); `a29c_565.py`: 18 more are the top halves of 400x480 both screen captures (17 Nookipedia, 1 Fandom S#13706); total 737 = 721 + 16 exactly. The body now says that 18 of the 737 are top halves. `a29b_565.py`: 8 more 400x240 PNGs have at most 32/64/32 levels per channel but another expansion rule (S#2733, 2924 to 2927, 3125, 9545, 13783); not counted | `audit a29_ids.json` |
| 14 | Album JPEG: 2,898 of 3,292 native NL JPEG originals carry the full IJG quality 80 tables, 2,875 of them 4:2:2 | CONFIRMED exactly | `a30_jpeg.py`: reference tables from Pillow's own quality 80 encode, both luma and chroma tables must match, `get_sampling` for 4:2:2: 3,292 / 2,898 / 2,875 | Fandom originals at 400x240 |
| 15 | Over all 400x240 items 3,445 quality 80 files of 3,902 JPEGs | CORRECTED (small) | old 3,445 of 3,902; new 3,438 of 3,900 when items are selected by catalogue size (`a30_jpeg.py`), 3,438 of 4,180 when selected by the actual pixel size of the loaded file (2,898 Fandom + 540 of 888 other JPEGs). The q80 count is rule stable within 7 files; the denominator is not. Body now reads "about 3,440 quality 80 files of 3,900 to 4,180 JPEGs, depending on the selection rule" | all 400x240 items |
| 16 | NL zenith #2120E7 on lossless frames | CONFIRMED | `a07_zenith.py`: modal exact colour of row 0 on the 81 lossless frames with a blue top row: #005DC6 44 (encyclopedia title bar, not sky; see 16.2 variants), #2120E7 18, the rest scattered; #2120E7 is the only recurring deep sky value | lossless set |
| 17 | NL zenith #2121E5 on album JPEGs | CONFIRMED | same script, Q80 Fandom originals with a blue row 0: rounded row 0 median #2121E5 on 32 of 108 frames (next #212DE5, 7); the 41 deep frames (G under 40) have median #2121E5 | Q80 originals |
| 18 | NL walking strip: top #2040F0 to #2050ED, last sky rows #2166FA to #2172FD | CONFIRMED with a wider spread | `a08_strip.py` on the 17 walking frames of item 1: row 0 #2138EF to #2949F7 on 12 frames, lighter #3753F7 to #6C72F0 on 5 (cloud or haze in row 0); end rows #2168F8 to #2971FF on 13, lighter on 4 | S#1341, 1351, 2657, 2694, 9409, 10507, 10678, 10932, 11143, 13257 and others |
| 19 | NL night zenith #01215A, flat over rows 0 to 79 | CONFIRMED; evidence label CORRECTED | `a13_night.py`: 10 row band medians #00205A in all bands on S#12601 and #01215A to row 49 on S#13760; tally over 65 dark uniform 400x240 top rows (stricter than V2's 114): #01215A 18, #403D6A 5, #072559 4, #323663 4. Correction: S#12601 is an 807x484 PNG (about 2x, resampled), not a Q80 JPEG; S#13760 is Q80. Its moon radii are in 807 px units | S#12601, 13760 |
| 20 | NL haze, far over near luminance median 2.15 (range 1.61 to 3.19), over the 100 rows below the horizon; saturation 0.81 to 0.97 near, 0.41 to 0.54 far | CONFIRMED | `a14_haze.py` on the walking frames of item 1 (third sample, 15 usable, S#1341 and 11143 are the same frame): far band rows h+2 to h+21, near rows 170 to 236, green mask: median 2.25 (1.21 to 2.67; 1.21 is S#9207, a pale winter ground); the band stays over 1.1 x near for 72 to 142 rows below the horizon (median about 102); saturation 0.80 to 0.96 near, 0.38 to 0.54 far on the green town frames. Arithmetic: 240 minus 35 = 205 ground rows | S#1341, 1351, 2657, 2694, 2697, 9207, 9409, 10317, 10507, 10638, 10649, 10678, 13257, 13617 |
| 21 | NL haze far colour: median #88BF68, per frame #B5E36B (S#2657), #B8E168 (2697), #BDE56B (10507), #ABD570 (13257) | CONFIRMED | same script: per frame far medians identical for S#2657, 2697, 10507 and within 1 level for S#13257 (#ACD470); median over 15 frames #A0C66B, inside the stated #88BF68 to #B5E36B target range | as item 20 |
| 22 | NL spring and summer near grass #026637, #1E7542, #107641, #0C6C3A, #0D7541, #06723F | CONFIRMED except the light end, CORRECTED | `a15_grass.py` plus a plain mask run (G over R and B, G over 60, rows 170 to 236): S#1246 #026637 (exact), 1245 #107442 (old #1E7542, R 14 levels lower), 1341 #067140, 9788 #056939, 13257 #0C7540, 10649 #05723F, hue 150 to 153 on all six. With a hue 120 to 170 mask all six are #026538 to #077240. The near grass range is therefore #026637 to #107442 (16.3 table and 16.11 updated) | S#1246, 1245, 1341, 9788, 13257, 10649 |
| 23 | CF day grass #0A8C54 (hue 154, 240 frames) | CONFIRMED | `a17_cfgrass.py` (third mask: G over R+20, hue 120 to 175, max channel over 107, whole frame): #0A915B, hue 156, IQR #038858 to #219764, 164 frames (832x456 #099261, 640x480 #0A9159); within 7 levels | CF 832x456 and 640x480 frames |
| 24 | NL dry sand #EFB67B (S#1539, LL) | CONFIRMED | `a18_sand.py` (sand mask R over G over B, R minus B over 50, rows 60 to 169 away from water): median #F7BE84, 8 levels lighter, upper luminance mode #FFD394 | S#1539 |
| 25 | NL wet band luminance 0.62 to 0.71 of dry | CONFIRMED | same script, wet = sand hue within 12 px of water and under 0.85 of dry L: 0.63, 0.67, 0.63 against the lit dry mode on S#1539, 12286, 13496 (0.67 to 0.78 against a dry median that includes palm shadows; state the reference) | S#1539, 12286, 13496 |
| 26 | Foam line #ECF2F8 / #DBDDF6 | CONFIRMED | same script, min channel over 205 and B at least R, within 12 px of water: #DEF7FF, #EFF8FD, #E1E9FC: a blue white, never pure white | S#1539, 12286, 13496 |
| 27 | NL day snow #A1A6E0 (S#3087) | CONFIRMED | same script, rows 110 to 239, x 150 to 329, bluish non white mask: #A0A4DC (p10 #9497CD, p90 #B1B1E7) | S#3087 |
| 28 | NL sea, lossless #3169AD (S#9174); use median #3A76C4 | CONFIRMED (hue, R, G); B lower on this sample | same script, B over R+60: S#9174 #296DAD (p10 #2165AD, p90 #4A86D6); S#2646 #3171AD, 2648 #3975BD, 1337 #397BB6, 12286 #4578AC, 13496 #4886C0, 13255 #2E6EB4; the median of these seven is #3975B4 [S], 16 levels lower in B than #3A76C4. Keep #3A76C4 for open sea in sun and #3169AD to #3975BD for lossless calm water | S#9174, 2646, 2648, 1337, 12286, 13496, 13255 |
| 29 | NL night grade, mid tones (0.45, 0.44, 0.75), linear (0.17, 0.16, 0.53); whites (0.26, 0.25, 0.65) roll off further | CORRECTED | Pair S#12062 (day) and S#12036 (night), both about 807x484 PNG; best integer shift (0, 0), correlation 0.70. `a24_grade.py` flat pixels (3x3 range under 25), day values 50 to 220: (0.36, 0.32, 0.68); `a24d_quant.py` percentile matching inside boxes on the same surfaces (immune to misalignment): fence (0.30 to 0.34, 0.29 to 0.32, 0.68 to 0.73), yellow flowers (0.43, 0.43, 0.77), path (0.28, 0.28, 0.69). New value (0.34, 0.31, 0.70) [S mean of the flat pixel value and the median of the four box p50 values, (0.32, 0.31, 0.71)], range by surface R 0.28 to 0.45, B 0.68 to 0.77; linear (0.09, 0.08, 0.46). V3's (0.45, 0.44, 0.75) is the light end (flowers). Whites: the p90 ratios are 0.44 to 0.57 in R and G, LIGHTER than the medians, so "highlights roll off further" is not reproduced and is removed. Arithmetic check of the old linear values: 0.45^2.2 = 0.173, 0.44^2.2 = 0.164, 0.75^2.2 = 0.531 (correct as arithmetic) | S#12062, 12036 |
| 30 | NL dusk grade midpoint (0.83, 0.58, 0.67), range R 0.70 to 0.96, G 0.48 to 0.69, B 0.59 to 0.75 | CONFIRMED | `a24_grade.py` against day S#11985 (shift 4 to 5 px down, 1 to 2 right): flat mid tones (0.88 to 0.89, 0.60, 0.65 to 0.66) on S#12000, 12005, 12015; whites (0.71 to 0.73, 0.51, 0.65); all inside the stated ranges. Midpoints: (0.70+0.96)/2 = 0.83, (0.48+0.69)/2 = 0.585, (0.59+0.75)/2 = 0.67 | S#11985, 12000, 12005, 12015 |
| 31 | NL interior ambient (80, 70, 166) #5046A6; lamps add +97 to +99 per lit channel at Level 1; Level 2 doubles, Level 3 adds half a step, Level 4 equals Level 3 | CONFIRMED | `a26_lamps.py`, whole frame method (median of lit minus off over unclipped room pixels): Level 1 red (+86, 0, 0), indigo (+82, 0, +57), orange (+86, +41, 0), yellow (+86, +84, 0), light blue (0, +39, +58): untouched channels exactly 0 (additive, not a multiply); the whole room average is about 0.88 of the wall box value, matching the floor 0.84 rule. Unclipped G series orange 41, 78, 97, 97 and light blue 39, 78, 97, 97 give 1, 1.9 to 2.0, 2.4 to 2.5, 2.4 to 2.5 times Level 1. Ambient in a new box x 330 to 394, y 80 to 189: (72, 63, 150), so the off value falls off with position. Arithmetic note: "0.080 to 0.445" uses the sRGB curve; with the 2.2 power stated in the preamble it is 0.078 to 0.453 (body now says so) | S#12260, 12639 to 12642, 11052 to 11055, 12315 to 12318, 13738 to 13741, 11225 to 11228 |
| 32 | NL player blob on open sand (0.55, 0.55, 0.67), luminance 0.56 | CONFIRMED | `a27_blob.py` second method (darkest 25% of sand pixels inside a hand placed ellipse under the feet, reference = sand in an annulus at 1.3 to 1.8 radii): (0.55, 0.55, 0.62), luminance 0.56. A box at the blob edge gives 0.68, so the value needs the core | S#1539 |
| 33 | NL player blob indoors (0.52, 0.52, 0.59) | CORRECTED | old (0.52, 0.52, 0.59) or (0.48, 0.49, 0.57); new (0.62, 0.58, 0.67), luminance 0.61. Method: same ellipse script (0.62, 0.59, 0.69) and a row dump at rows 188 to 196: core #847DA5 to #8C8AAD against the floor #D6D7F7 to #DED7F7 beside it on the same rows. The old values divide by a brighter floor (about #FFF7FF) further back; the floor falls off toward the camera, so the reference must be the same row | S#2204 |
| 34 | CF player blob on snow, core 0.55 to 0.64, tint (0.53, 0.53, 0.58) | CONFIRMED | same ellipse script on snow pixels: (0.56, 0.56, 0.61), luminance 0.57 | S#111 |
| 35 | Grain, lossless medians: all 737 frames 2.78 (2.09 to 3.75), 313 house rooms 3.83, outdoor frames with grass over 0.2: 2.56 (IQR 2.19 to 3.03, 75 frames) | CONFIRMED | `a35_grain.py` (own 9 neighbour sum, border dropped): all 737: 2.78 (IQR 2.09 to 3.75); rooms 3.83 (IQR 3.10 to 4.80); outdoor by my own grass rule 2.56 (IQR 2.14 to 2.97, 84 frames), 2.61 without a balloon (39). Target arithmetic: 2.56 minus about 0.20 = 2.36, stated as 2.35; cross check 2.66 minus 0.14 minus 0.20 = 2.32. The 565 step allowance (0.15 to 0.26) was not re measured (no clean render in this audit) | lossless set, `V2/house_ids.npy` |
| 36 | CF flooring: 94 of 94 at 256x256 mirror symmetric in both axes; colours median 53 (18 at 16 or fewer, 1 at 256); 88 of 94 on the RGB565 grid; grain median 3.6 | CONFIRMED, wording CORRECTED | `a36_floor.py` (own selection from the "City Folk flooring textures" category, 94 items, same set as R2): mirror 94 of 94 (array equals both flips exactly), colours median 53, 18 at 16 or fewer, 1 at 256, grain 3.59. `a36b_grid.py`: 88 of 94 lie on the ROUNDED 565 grid (round(q x 255/31) and round(q x 255/63)); only 4 lie on the bit replication grid that defines the lossless frames, and all 94 have at most 32/64/32 levels. Body now names the rounded grid | S#5681 and the 93 other flooring textures |
| 37 | NL cumulus on S#13187: 30.4% of rows 0 to 139, median #CCE1FE, p10 #A6C5FC, p90 #DDEAFD, maximum (232, 250, 255) | CONFIRMED (colour); cover threshold dependent | `a37_misc.py` (cloud = min channel over 150 and B over 200): 35.6% of rows 0 to 139, median #CBE0FE, p10 #ACC9FB, p90 #DBEAFF, maximum (245, 250, 255), still never pure white; sky beside them #2239ED | S#13187 |
| 38 | NL late winter grass #5A8548, #5B8D3B, #659844 (hue 90 to 102) | CONFIRMED | same script, near band plain green mask: S#1433 #62854A, 1591 #5D8D3B, 1351 #679845, hue 95 to 96 | S#1433, 1591, 1351 |

**Tally.** 38 numbers audited: 32 CONFIRMED as values (5 of them with a label, range or wording fix: items 2, 10, 13,
19, 36) and 6 CORRECTED as values (items 3, 8, 15, 22, 29, 33). Corrections are applied in the body, in 16.10 and in 16.11.

**Arithmetic rechecked [audit].** Horizon fractions 34/240 = 0.142, 35/240 = 0.146, 43/240 = 0.179, 46/240 = 0.192.
Horizon ray: (0.5 minus 0.144) x 2 x tan 14 deg = 0.1775, 10.07 deg above the axis, 35 minus 10.07 = 24.9 deg below
horizontal; the tangency of a ray with a parabolic bend gives R = 2(H minus z0 tan a)/tan^2 a (derived again here), so
2(3.6 minus 4.1 x 0.4648)/0.2160 = 15.7 P, with 3.5 and 4.15 it is 14.5 P, and the press 0.165 row gives 14.4 P: all
CONFIRMED. Fish lengths in bobbers 10.4/9 = 1.16, 22.8/10 = 2.28, 24/10 = 2.4, 38/8 = 4.75, 44/7 = 6.29: CONFIRMED;
lengths in P recomputed with 7.65 bobbers per P (item 3). Linear values: 0.50, 0.63, 0.68 to the 2.2 power give 0.218,
0.362, 0.428 (CONFIRMED); grass blob 0.41 to 0.50 gives 0.14 to 0.22, not "about 0.19" (16.10 fixed). Dusk midpoints
CONFIRMED (item 30). Lamp linear range uses the sRGB curve (item 31, body fixed). Counts: 16,942 minus 3 = 16,939, minus
2 non images = 16,937; the 16.0 table rows sum to 16,937 because three rows are image counts, not record counts (note
added); first pass layouts summed to 16,936 because one "Wii native" item was missing (added); Fandom original formats
6,241 + 2,537 + 37 + 1 = 8,816 CONFIRMED; 721 + 16 = 737 CONFIRMED.

**Internal consistency fixes [audit].** 16.10 now matches the body for the NL night window range (#F1C791 to
#FCD45E; #F4D08C was one frame, S#9215, in V2), the walking sky strip (body values instead of the V2 summary #2044F2 to
#216BFD, in 16.10 and 16.11), the interior blob, the night grade and the fish length range; the 16.10 fire core
#FFDE8E is now traced in the body (R2, S#4854 render). 16.11 rows for sky, grass, night, blobs, fish, album JPEG count and
CF interior share follow the corrected body. Other 16.10 values (grass, path, sand, snow, cliff, water, canopy, trunk,
interior light model, skin, hair, clothing, clouds, UI) were checked against the body and match.

**Labels [audit].** S#12601 is an 807x484 PNG, not Q80; the night day pair S#12036 and 12062 and the dusk set S#11985 to
12015 are also about 807x484 PNGs (Fandom originals of a 2x capture), fine for ratios. S#622 in the CF look up horizon list
is a promo frame (P), now marked. Nintendo Life items S#16412 to 16494 are cited six times as if retail; the 16.0 note
now says their retail status cannot be confirmed. The CF house look up set is mostly 420x308 crops, not 832x456. The night
grade confidence drops to M and the whites claim to L.

**Writing rules [audit].** No dash characters (U+2010 to U+2015, U+2212) anywhere; the only hyphen characters in the file
are the Markdown table rules. No angle brackets were added. Every number added in the audit cites a script in
`verify_shots/AUDIT/` and the S# IDs it ran on.

**Not re measured in this audit:** the 565 step allowance (0.15 to 0.26) in the grain target, the moon and lamp glow radii,
the fish bobber sizes (`a38_bobber.py` red masks give 8x9, 7x8 and 9x9 on S#12899 to 12901, the red half only, so the
listed 10x9 and 10x8 cannot be confirmed or refuted), the CF night and rain multipliers, the clothing and icon facts.

### 16.14 Validation in the test scene (`style_test_town/town_river_test.py`, QA printed by `--render`)

Section 16 values applied to the town river scene and measured on its renders (1200x720, resized BILINEAR to 400x240
where a 240 row figure is quoted):
* Camera solved from the 16.1 targets (player share 0.32, feet row 0.60, inside the retail 44 to 63%): height 3.40 P,
  feet distance 4.37 P; bend radius calibrated to the 0.144 horizon row: 13.28 P (16.1 arithmetic said 14.5 to 15.7 P
  with the 12.1 camera; the different feet row explains the gap).
* Haze (ground and water only): far band 2 to 22 rows below the horizon over the 170 to 236 band, on a ground only
  render pass: 2.12 (retail median 2.15). Applying the same haze to trees and props washed them to flat lime, which no
  capture shows, so haze stays a ground and water layer (16.10 H applies to those materials only). Confirmed by direct
  measurement of objects at the horizon in 16.17.
* River lines p95 over median 1.25 (16.4 rivers 1.28).
* Grain: grass pixels only 2.79 against 2.87 (p25 2.31, p75 3.40) on 85 lossless retail frames with over 25% grass;
  whole frame 1.93 against 2.92 on the same frames. The whole frame gap comes from larger smooth water and sky areas
  and fewer objects than a typical retail frame, not from the grass. Raising the print density further turned the dots
  into carpet like noise, which the captures do not show. Compare grain on matching content.
* Colour envelope: the retail walking sky strip itself counts as near neon (S about 0.87, V about 0.94); retail frames
  with over 8% sky give a neon share median 0.3%, p75 3.2%, p90 9.0% (2,284 frames); the render's 9.8% sits at p90
  because its sky strip is unobstructed. Treat the 2% envelope (7.1) as a ground and object rule, not a sky rule.
* Night and dusk water: grading the day water colour leaves it lavender; use the measured time of day water colours
  (dusk #3C5596, night #0E1D53) as the base and haze toward the same family. Render medians: dusk #40558F, night #132265.

### 16.15 Test scene revisions after review (user direction)

* Haze returned to the ORIGINAL test scene setting at the user's request: ground only, from halfway between the
  player's feet and the visible ground edge to the edge, mixed up to 0.8 toward #9FCE60 (the user's retail capture far
  ground); no haze on water. Ground only pass ratio 1.26 (the 16.3 tuned variant reached 2.12; both are documented).
* Object shadows follow the object's profile: trees, bushes and flower clumps cast a decal shaped like their top down
  silhouette (the user's observation of the captures: player and rock shadows may be round, but trees and flowers
  match the feature's outline). Method: rasterise the object's triangles projected straight down, testing leaf and
  petal card texture alpha at 0.5, grow the footprint 3 px and blur 3 px (160 px mask), multiply decal with the
  canopy or contact values, canopy darker toward the centre (35%). Decals must be subdivided (16 x 16): a single large
  flat quad sinks under the rolling log ground in its middle (sagitta L^2 / 8R, about 0.05 P for 2.2 P at R 13 P)
  and disappears. Render QA: flower clump south edge 0.78 (target 0.68), near tree south edge 0.67 (target 0.77 to
  0.84 at the rim, darker inside). Player and rock keep round blobs.
* Haze nudge after the darker 16.3 grass (user): colour #8EC668 (between the measured retail far ground #88BF68 and
  #94CC78) instead of the lime #9FCE60, start at 0.4 of the way from the feet to the visible ground edge (was 0.5),
  maximum mix 0.9 (was 0.8). Ground only pass ratio 1.38 (was 1.26).

### 16.16 Second test scene: a plaza in front of a village hall (`style_test_town/town_hall_test.py`)

Original hall and villagers built in the series' design language (squat chunky building, soft bulging eaves, a central
clock bay, stone columns, plank walls, cobble plaza; round headed villagers with stub limbs, section 5), with sizes,
framing and colours measured on retail frames. Revision 2 follows user review (plants on cobbles, texture mismatch,
boxy hall, poor villagers, odd haze on the plaza).

* Framing in front of a building [by eye on 3x and 5x NEAREST crops, M]. The walking view (feet row 0.60) cannot show
  a building: with the horizon at 0.144 a wall 1 P tall already reaches row 0.0. Retail frames in front of buildings
  put the player low: S#2855 (town hall exterior) feet row 220 of 240, hat top 138; S#1436, S#1441 feet 214 to 215, cap
  top 134 to 135. Used: feet row 0.90, player share 0.33.
* Buildings and props lean back about 16 deg; characters stay upright [S arithmetic on fits, M, one frame each].
  Building verticals in S#2855 barely converge: least squares on the strongest horizontal gradient per row gives the
  outer wall corners a drift of 5.7 and 7.7 native px between rows 120 and 195. With the camera solved for an upright
  player (height 3.58 P, distance 3.42 P) upright walls would drift 16.0 px, walls leaned back 14 deg 7.7 px, 18 deg
  5.8 px. The ground still fits pitch 35, so the objects are tilted away: a shear y += z tan(16 deg) on buildings and
  props before the rolling log bend. Characters are NOT leaned: the player's head width over figure height reads
  about 0.54 in S#2855 and 0.53 in the upright river render, but 0.44 when the same shear is applied (the first
  revision did this and the user saw a squeezed player). Mechanism in the game: cannot confirm. With this camera the
  S#2855 proportions give a hall eave of about 1.1 P and a clock gable top of about 1.85 P (the hall is modelled at
  1.32 P eaves and scaled by 0.8).
* Hall colours from S#2855 (median boxes by eye, M): plank wall #AB854A (p10 #866936, p90 #DBB264), lavender grey stone
  #A08998 to #E6C0BD lit, window panes up to #F1EEC0, cobbles #928498 (this frame) to #ADB2C6 (S#3152). Roof kept at the
  NL house roof family #BB3F5B (16.7).
* Villagers: one head mesh per villager; the muzzle and cheeks are Gaussian bumps of the head surface and the ears are
  extruded from head faces (2 to 4 steps, flattened, tapered), with an ear texture holding a fur border and a pink inner
  panel, so nothing reads as stuck on. 16 segments around the head (5.2: 12 to 16 per turn). 816 triangles per
  villager (the 5.2 arithmetic estimate is 500 to 800, L).
* Flowers: beds of cup tulips (8 sided lathe, alternate rim vertices raised into petal tips) on stems with broad curved
  leaves, and flat five lobed pansies tilted toward the viewer with a dark blotch and a yellow eye, after S#2855 and
  S#1436 by eye; profile shadows from their top down footprint.
* Soft geometry: every box is a rounded box (cube grid pushed onto a box with radius 0.04 to 0.075 P edges, smooth
  shaded), columns are 12 sided lathes with cushion bases and rolled capitals, the roof is two thick rounded slabs with a
  ridge roll, the clock gable is an arch topped slab with a torus trim (a bent tube with a fixed frame: polyline tubes
  twist on arcs).
* Haze: first applied to every object at the user's suggestion; withdrawn after measurement (16.17): retail objects at
  the horizon are not hazed. The weed reading that seemed to support object haze (S#13257, #44964D near, #9DCE6D far)
  was a mask artefact: the green mask also caught hazed ground; the far weeds themselves are saturated. Current: ground
  only, anchored to ground places, cobbles toward #C6C3D6.
* Texture density: the user found the hall and characters cleaner than the cobbles. All textures now share one painted
  treatment: about 64 to 80 texels per P (planks 64, cobbles 80, fur 64; faces about 110, inside the 80 to 150 texels per
  P that section 6's CF face reading implies), bilinear, fbm mottle 4 to 7% plus felt grain; the shared player textures
  get the same pass (face resampled to 0.75).
* Plants only on grass: the build asserts zero plaza weight under every tree, bush and flower bed; flowers on the hall
  sit in window boxes.
* Render QA (day): player share 0.33, villagers 0.33; clock gable top row 0.011, eave row 0.25 (S#2855: 0.067,
  0.417, with the hall right behind the player; here it stands about 1 P further back to leave room for the
  villagers). Neon 4.6% (more sky and the tulips), near black 0%, near white 1.9%. Player blob on cobbles, render with and without the blob
  divided per pixel: median 0.55 over 4,205 shaded pixels (the configured multiply; section 3 grass range 0.41 to
  0.50). Triangles: one villager 816, scene about 122,000.

### 16.17 Haze review (after the town hall scene; measurements on retail frames, scripts in the session scratchpad)

What retail NL hazes and what it does not:

| element at the horizon | far | near (same frame) | hazed? |
|---|---|---|---|
| grass, S#1246 rows 36 to 48 beside a far trunk | #A0CF68, L 181, S 0.50 | #026738, L 67, S 0.98 (rows 200 to 235) | yes, x2.7 |
| tree trunk standing on that horizon, S#1246 rows 8 to 34 | #985A33, L 104, S 0.66 | (a nearer cherry trunk #AF7929) | no |
| cedar crown at the horizon, S#1246 | #093219, L 35, S 0.82 | | no (darker than any hazed ground) |
| cherry crown at the horizon, S#1246 | #F585CC, L 175, S 0.46 | #F492CD, L 182, S 0.40 | no (same colour) |
| weeds on the far ground, S#13257 | saturated green (by eye on a 4x crop) | | no |
| cobbles, S#3152 (lossless) | #C6C3D6, L 198 | #ADB2C6, L 179 | yes, x1.11, toward lavender grey |

Rules (H for the ground, M for objects on 2 frames and 5 objects):
1. Haze is a ground and water material layer, not an atmosphere: objects standing in the hazed band keep their own
   colour, which is why far crowns and trunks read crisply against a pale far ground in retail frames.
2. The far colour depends on the surface: grass toward #88BF68 (16.3), cobbles toward a lighter lavender grey, sea
   toward #95B3F2 (16.4). A single lime target on paved ground is wrong.
3. Distances belong to ground places. A scene with a different framing (the town hall camera: 3.58 P high, 3.42 P
   back) maps the walking camera's haze distances (7.72 to 10.99 P view distance) through the ground points they hit:
   6.99 to 10.23 P from its own camera.

Contradictions resolved in this review: section 2 (retail unproven), 12.1 (press values only), 12.2 step 8 (retail off),
the 13 open question on retail haze, 16.10 (no rule for paved ground or objects), 16.14 against 16.16 (objects). All now
point here.

Strength in the test scenes: the river scene keeps the user's chosen setting (start at 0.4 of the way from the feet to
the visible ground edge, mix up to 0.9 toward #8EC668): far band ratio 1.38 against the retail median 2.15 (16.15). The
town hall scene uses the same ground places; measured by rendering with and without the ground haze and dividing per
pixel: haze touches 5.45% of the frame (far grass beside the hall), grass median 1.57, p95 2.42 (the far band reaches
the retail level), cobbles 1.04 (they lie close to the camera), objects unchanged (0 changed pixels outside the
ground by construction).

