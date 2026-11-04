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
| distance haze, NL press builds | ground brightens and yellows over the far 40 to 50% of visible ground: NL#953 L 99 to 104 near to 182 at the horizon; far colour typical #94CC78 (max #AAE677 NL#956); independent of sky colour; water too (NL#957 near #3B6CAC, far #96B8F9). Horizon nearly dissolves (sky L 191 vs far grass 177 to 197) | A, vA (4 press images). Retail NL haze UNPROVEN (no open far ground in NL#938 to 940). M |
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
| trees and props | NL#946 soft dark patch under trunk base, ratio about 0.7, irregular; NL#951 rock contact band (by eye) | cannot confirm |

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
| NL retail / near retail | deep blue #2135E7 (h 234, S 0.86, V 0.91) on retail NL#940; #242CE6 on NL#939 (Nintendo Direct 2012) | lower sky #4078DA to #6894D9 near a cloud (NL#939) | [S] `synth/sky_retail.py`, vC; [Au] NL#940 #2139E8, NL#939 #2323EF (strict blue mask): CONFIRMED within 9 levels. M (two small frames). Use the deep blue for an NL retail look; cyan only for an "early NL" look |
| NL snow night, evening | snow #4E56B1 under night light (NL#961); evening sky #7942FE (NL#964) | | vA. M |

### 4.2 Ground and grass

| role | CF | NL | source |
|---|---|---|---|
| grass | #009758 (CF#882), #009158 (CF#888), #0A8564 (CF#873): hue 154 to 166, V 0.47 to 0.57, teal green | press builds #188050 / #147D4D (hue 145 to 154); RETAIL emote captures lime: median #92DB10 (hue 81 to 82, S 0.93, V 0.86), mottle from p25 #82CC09 to p75 #A1E917 (NL#941, 942, 944, 945, all grass pixels incl. the blob shadow) [Au]; v2 #82CD0C is the dark quartile, CORRECTED | A, vA, vC, vD, Au. Season or build behind the lime cannot be confirmed. CF [Au]: CF#882 #009658, 888 #009158, 873 #068464, 878 #009B59: CONFIRMED. NL press [Au] #127C4B to #1D8551 (NL#946, 953, 951): CONFIRMED. M |
| far grass | same as near (no haze) | haze to #94CC78 (press) | A, vA |
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
| opacity | opaque, no visible bottom | transparent enough to show pier posts and tyres (NL#957, 967, press) |
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
| haze | none | strong in press builds, toward #94CC78 |
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
8. Haze (NL early look only): Camera Data View Z Depth into Map Range (near half of visible ground 0, far edge 1)
   then smoothstep, mix ground and water toward #94CC78 up to 0.8. CF and NL retail: off.
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
| retail NL outdoor look: haze present? grass lime (#92DB10) or teal green? sky always deep blue? | three of the strongest NL cues rest on press builds | 5 to 10 native 400x240 retail captures of open outdoor ground in summer and other seasons |
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
