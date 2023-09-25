"""
NOTE: carEntityWrds come from data/kaggle/usa_cars/USA_cars_datasets.csv

****************MOST IMPORTANT Guidelines to MUST follow*************
(1) The format is <entityLbl><entityWrd> followed by space. The following is
    NOT ALLOWED: <entityLbl><entityWrd><entityLbl><entityWrd> and it should be
    rewritten as follows: <entityLbl><entityWrd> <entityLbl><entityWrd>
    Therefore, <units_price_$><$><price><5000> is NOT ALLOWED because the code
    cannot assign two labels (i.e. unit_price and price) to one word
    (i.e. $5000). ***NOTE*** that the user will type $5000, but this is not a
    problem because the filtering code
    (i.e. Utilities.py:userIn_filter_splitWords()) will separate
    "$5000" into "$" "5000" and the NN will never see $5000

(2) When using range2 labels (e.g. "-", "to", "through"), the words
    before-and-after MUST be numbers and optionally their units. Following
    are some examples:
    "2015 - 2020"     " 2015 to 2020"         " 2015 through 2020"

    "$5000 - $9000"  "$5000 - 9000"  "5000 - $9000"  "5000$ - 9000$"
    "5000 - 9000$"   "5000$ - 9000"  "5000 - 9000"

    "miles 5000 - miles 9000"  "miles 5000 - 9000"  "5000 - miles 9000"
    "5000 miles - 9000 miles"  "5000  - 9000 miles" "5000 miles - 9000"




Problems:
---------
(1) Pre-tokenization
A string can be represented with single- or double-quotes. Following are
solutions to three cases where the characters within the string have
single- and double-quotes:
(i) If a string has single-quotes then it should be represented with
     double-quotes, otherwise python-compiler will give Syntax Error.
     Example: "price of 'rice' in china"
(ii) If a string has double-quotes then it should be represented with
      single-quotes. Example: 'price of "rice" in china'
(iii) If a string has both single- and double-quotes, and it is
       represented with single-quotes, then within the string all single
       quotes must be preceeded by an escape character forward-slash.
       Example: 'price "of" \'rice\' in china'

Valid formats of Placeholders:
------------------------------
e.g. entity => entityLbl or entityWrd;
     entityLbl = "color"; entityWrd = "red"

***NOTE: that <> means that code will fill this placeholder with a string
            or integer based on the context

Formats follow:
---------------
(1) <entityLbl><>
Code picks entityWrd belonging to entityLbl

(2) <entityLbl><entityWrd>

Some notes:
-----------
+ Instead of using a -ve, a previous range can be removed by
    mentioning a new range

NN must learn when to label or not-label the entityWrds:
--------------------------------------------------------
Add to segments certain entityWrds (e.g. between, to, through) that the NN must learn to NOT label. Example:
"i have to tell you that <year><> <range2><> <year><>   <mileage><>   <range2><-> <mileage><>  <units_mileage_mi><>  <units_price_$><> <price><>   <range2><-> <price><>  is between a few cars i want through"
In the above sentence, the NN must NOT label the "to", "between", "through"


Segments that are NOT ALLOWED:
------------------------------
" <price><>   <units_price_$><>  <range2><-> <price><> ",
" <price><>   <range2><->  <units_price_$><> <price><> ",
Reason: if "units_price_$" is "dollar" then its typo could become "dolkar". The
function Utilities.py: userIn_filter_splitWords(userIn) removes the hyphen
because "dolkar" is neither a number or a unit
"""
import random

PLACEHOLDER_ID_START = "<"
PLACEHOLDER_ID_END = ">"

train_sentences = ()
val_sentences = (
    # following is how the user should normally type

    # NN must sometimes label entityWrds as "O"; usually (but not always) when they are followed by an "O" word
    "<color><>  <other><to> <model><>  <other><between>  <other><$>  <other><> <less><>  <other><than>   <units_price_$><> <price><> ",
    " <other><larger>  <other><dollars>   <other><>  <mileage><>   <range2><-> <mileage><>  <units_mileage_mi><>  <other><above>  <other><>  <year><> <range2><> <year><>  <other><range>  <other><> <units_price_$><> <price><>   <range2><-> <price><>  <other><remove> ",
    " <other><smaller>  <other><>  <other><__year>  <other><higher>  <other><>  <price><> <units_price_$><> <color><>   <other><below>   <brand><> ",
    "  <brand><>  <other><through> <less><>  <other><than>  <units_price_$><>   <price><>  <other><greater>  <other><prices>   ",
    "  <other><mileages>  <other><>  <model><>   <other><little>  <other><>  <price><>   <range2><-> <price><>  <units_price_$><>   <other><less>  <other><>  <mileage><>   <units_mileage_mi><> ",
    "<price><> <units_price_$><>  <other><price> <other><>  <brand><>  ",
    "  <other><years> <price><>   <range2><-> <price><> <units_price_$><> <other><lower>  <other><dollar> <color><>  <other><mileage>  ",

    # NN must not always label 1970 - 2024 as belonging to year
    "<model><> <brand><>  <other><__year>   <price><__year> <range2><> <price><__year> <units_price_$><>   <other><>   <mileage><__year> <units_mileage_mi><miles>  <other><more> ",
    " <other><>  <mileage><__year> <units_mileage_mi><>   <other><and>   <price><__year> <units_price_$><> <price><__year>  <other><or>  <more><more>",
    "<year><__year> <color><>  <units_price_$><> <price><__year> <more><more> <mileage><__year> <units_mileage_mi><miles>",
    " <other><__year>  <other><> <mileage><__year> <units_mileage_mi><miles>   <less><less>  <other><than>   <units_price_$><> <price><__year>",
)
test_sentences = (
    # following is how the user should normally type

    # NN must sometimes label entityWrds as "O"; usually (but not always) when they are followed by an "O" word
    " <other><mileages> <brand><>  <other><through>  <brand><>  ",  # In Predict, NN must not label "through" as range2 label
    " <other><$>  <other><> <mileage><>   <range2><-> <mileage><>  <units_mileage_mi><>  <units_price_$><> <price><>   <other><between>  <other><>  <range2><-> <price><> <other><to>  <model><> ",
    "<other><less>  <other><dollars>  <other><>   <other><__year>  <other><larger>   <color><>  <other><smaller> <brand><> ",
    " <other><under>   <brand><>  <brand><>  <other><mile>   <other><> <less><>  <other><than>   <units_price_$><> <price><>",
    " <model><>   <other><mi>   <brand><>  <other><price>   <color><>  <less><>  <other><than>  <mileage><>   <units_mileage_mi><> ",
    "the range of prices is above or below but i say <price><>   <range2><-> <price><> <units_price_$><> which is neither greater or little",

    # NN must not always label 1970 - 2024 as belonging to year
    "<year><__year> <model><> <less><less>  <other><than>  <mileage><__year> <units_mileage_mi><> <color><> <price><__year> <range2><> <price><__year> <units_price_$><>",
    "  <other><__year>  <other><> <units_price_$><> <price><__year> <more><more> <mileage><__year> <units_mileage_mi><>",
    " <units_price_$><> <price><> <range2><-> <price><>  <units_price_$><> <price><__year> <mileage><__year> <units_mileage_mi><> <more><> ",
)
full_sentences = {
    "train": train_sentences,
    "val": val_sentences,
    "test": test_sentences
}

brand_segments = (
    # neural-net must memorize, e.g., toyota is brand, without any hints
    "<other><> "
    "<brand><>  ",
    "<other><> <brand><>  ",
    "<brand><>  <other><> ",
    "<other><> <brand><>  <other><> ",
)
model_segments = (
    # neural-net must memorize, e.g., camry is model, without any hints
    "<model><>  ",
    "<other><> <model><>  ",
    "<model><>  <other><> ",
    "<other><> <model><>  <other><> ",
)
color_segments = (
    # neural-net must memorize, e.g., red is color, without any hints
    "<color><>  ",
    "<other><> <color><>  ",
    "<color><>  <other><> ",
    "<other><> <color><>  <other><> ",
)
price_segments = (
    # a number, e.g. 2022 by itself does not tell whether user refers to price,
    # year, or mileage; so a hint is needed, for ,e.g., a $ sign

    # " <price><>",
    "<units_price_$><$> <price><>",
    "<other><> <units_price_$><$> <price><>",
    "<units_price_$><$> <price><> <other><>",
    " <other><> <units_price_$><$> <price><> <other><>",
    "<units_price_$><> <price><>",
    "<other><> <units_price_$><> <price><>",
    "<units_price_$><> <price><> <other><>",
    " <other><> <units_price_$><> <price><> <other><>",
    "<price><> <units_price_$><dollars>",
    "<other><> <price><> <units_price_$><dollars>",
    "<price><> <units_price_$><dollars> <other><>",
    " <other><> <price><> <units_price_$><dollars> <other><>",
    "<price><> <units_price_$><>",
    "<other><> <price><> <units_price_$><>",
    "<price><> <units_price_$><> <other><>",
    " <other><> <price><> <units_price_$><> <other><>",

    # "  <price><> <range2><-> <price><> ",
    " <units_price_$><> <price><>   <range2><> <price><> ",
    " <other><> <units_price_$><> <price><>   <range2><> <price><> ",
    " <units_price_$><> <price><>   <range2><> <price><> <other><>",
    " <other><> <units_price_$><> <price><>   <range2><> <price><> <other><>",
    " <price><>   <units_price_$><>  <range2><> <price><> ",
    "<other><> <price><>   <units_price_$><>  <range2><> <price><> ",
    " <price><>   <units_price_$><>  <range2><> <price><> <other><>",
    " <other><> <price><>   <units_price_$><>  <range2><> <price><> <other><>",
    " <price><>   <range2><>  <units_price_$><> <price><> ",
    " <other><> <price><>   <range2><>  <units_price_$><> <price><> ",
    " <price><>   <range2><>  <units_price_$><> <price><> <other><>",
    " <other><> <price><>   <range2><>  <units_price_$><> <price><> <other><>",
    " <price><>   <range2><> <price><>  <units_price_$><> ",
    " <other><> <price><>   <range2><> <price><>  <units_price_$><> ",
    " <price><>   <range2><> <price><>  <units_price_$><> <other><>",
    " <other><> <price><>   <range2><> <price><>  <units_price_$><> <other><>",

    # " <range1><>  <price><> <price><>  ",
    " <range1><>   <units_price_$><> <price><> <price><>  ",
    " <other><> <range1><>   <units_price_$><> <price><> <price><>  ",
    " <range1><>   <units_price_$><> <price><> <price><>  <other><>",
    " <other><> <range1><>   <units_price_$><> <price><> <price><>  <other><>",
    " <range1><>   <units_price_$><> <price><> <other><and> <price><>  ",
    "<other><> <range1><>   <units_price_$><> <price><> <other><and> <price><>  ",
    " <range1><>   <units_price_$><> <price><>  <other><and>  <price><> <other><> ",
    " <other><> <range1><>   <units_price_$><> <price><> <other><and> <price><>  <other><>",
    " <range1><>  <price><>  <units_price_$><> <price><>  ",
    " <other><> <range1><>  <price><>  <units_price_$><> <price><>  ",
    " <range1><>  <price><>  <units_price_$><> <price><>  <other><>",
    " <other><> <range1><>  <price><>  <units_price_$><> <price><>  <other><>",
    " <range1><>  <price><> <price><>   <units_price_$><> ",
    " <other><> <range1><>  <price><> <price><>   <units_price_$><> ",
    " <range1><>  <price><> <price><>   <units_price_$><> <other><>",
    "<other><> <range1><>  <price><> <price><>   <units_price_$><> <other><>",
    " <range1><>  <price><>  <other><and> <price><>   <units_price_$><> ",
    " <other><> <range1><>  <price><>  <other><and> <price><>   <units_price_$><> ",
    " <range1><>  <price><>  <other><and> <price><>   <units_price_$><> <other><>",
    "<other><> <range1><>  <price><> <other><and>  <price><>   <units_price_$><> <other><>",

    # " <less><>  <price><>  ",
    " <less><>   <units_price_$><> <price><>  ",
    " <other><> <less><>   <units_price_$><> <price><>  ",
    " <less><>   <units_price_$><> <price><>   <other><> ",
    "  <other><> <less><>   <units_price_$><> <price><>   <other><> ",
    " <less><> <other><than>  <units_price_$><> <price><>  ",
    "  <other><> <less><> <other><than>  <units_price_$><> <price><>  ",
    " <less><> <other><than>  <units_price_$><> <price><>   <other><> ",
    "  <other><> <less><> <other><than>  <units_price_$><> <price><>  <other><>  ",
    " <less><>  <price><>   <units_price_$><> ",
    "  <other><> <less><>  <price><>   <units_price_$><> ",
    " <less><>  <price><>   <units_price_$><>  <other><> ",
    "  <other><> <less><>  <price><>   <units_price_$><>  <other><> ",
    " <less><>  <other><than>   <price><>   <units_price_$><> ",
    "  <other><> <less><>  <other><than>   <price><>   <units_price_$><> ",
    " <less><>  <other><than>   <price><>   <units_price_$><>  <other><> ",
    "  <other><> <less><>  <other><than>   <price><>   <units_price_$><>  <other><> ",

    # "  <price><>  <less><> ",
    "  <units_price_$><>  <price><>  <less><> ",
    "  <other><>  <units_price_$><>  <price><>  <less><> ",
    "  <units_price_$><>  <price><>  <less><>  <other><> ",
    "   <other><> <units_price_$><>  <price><>  <less><>  <other><> ",
    "  <units_price_$><>  <price><>  <other><or>  <less><> ",
    "  <other><>  <units_price_$><>  <price><>  <other><or>  <less><> ",
    "  <units_price_$><>  <price><>  <other><or>  <less><>  <other><> ",
    "   <other><> <units_price_$><>  <price><>  <other><or>  <less><>  <other><> ",
    "  <price><>   <units_price_$><> <less><> ",
    "   <other><> <price><>   <units_price_$><> <less><> ",
    "  <price><>   <units_price_$><> <less><>  <other><> ",
    "   <other><> <price><>   <units_price_$><> <less><>  <other><> ",
    "  <price><>   <units_price_$><> <other><or> <less><> ",
    "  <other><>  <price><>   <units_price_$><> <other><or> <less><> ",
    "  <price><>   <units_price_$><> <other><or> <less><>  <other><> ",
    "  <other><>  <price><>   <units_price_$><> <other><or> <less><>  <other><> ",

    # " <more><>  <price><>  ",
    " <more><>   <units_price_$><> <price><>  ",
    " <other><> <more><>   <units_price_$><> <price><>  ",
    " <more><>   <units_price_$><> <price><>   <other><> ",
    "  <other><> <more><>   <units_price_$><> <price><>   <other><> ",
    " <more><> <other><than>  <units_price_$><> <price><>  ",
    "  <other><> <more><> <other><than>  <units_price_$><> <price><>  ",
    " <more><> <other><than>  <units_price_$><> <price><>   <other><> ",
    "  <other><> <more><> <other><than>  <units_price_$><> <price><>  <other><>  ",
    " <more><>  <price><>   <units_price_$><> ",
    "  <other><> <more><>  <price><>   <units_price_$><> ",
    " <more><>  <price><>   <units_price_$><>  <other><> ",
    "  <other><> <more><>  <price><>   <units_price_$><>  <other><> ",
    " <more><>  <other><than>   <price><>   <units_price_$><> ",
    "  <other><> <more><>  <other><than>   <price><>   <units_price_$><> ",
    " <more><>  <other><than>   <price><>   <units_price_$><>  <other><> ",
    "  <other><> <more><>  <other><than>   <price><>   <units_price_$><>  <other><> ",

    # "  <price><>  <>more<> ",
    "  <units_price_$><>  <price><>  <more><> ",
    "  <other><>  <units_price_$><>  <price><>  <more><> ",
    "  <units_price_$><>  <price><>  <more><>  <other><> ",
    "   <other><> <units_price_$><>  <price><>  <more><>  <other><> ",
    "  <units_price_$><>  <price><>  <other><or>  <more><> ",
    "  <other><>  <units_price_$><>  <price><>  <other><or>  <more><> ",
    "  <units_price_$><>  <price><>  <other><or>  <more><>  <other><> ",
    "   <other><> <units_price_$><>  <price><>  <other><or>  <more><>  <other><> ",
    "  <price><>   <units_price_$><> <more><> ",
    "   <other><> <price><>   <units_price_$><> <more><> ",
    "  <price><>   <units_price_$><> <more><>  <other><> ",
    "   <other><> <price><>   <units_price_$><> <more><>  <other><> ",
    "  <price><>   <units_price_$><> <other><or> <more><> ",
    "  <other><>  <price><>   <units_price_$><> <other><or> <more><> ",
    "  <price><>   <units_price_$><> <other><or> <more><>  <other><> ",
    "  <other><>  <price><>   <units_price_$><> <other><or> <more><>  <other><> ",
)
mileage_segments = (
    # a number, e.g. 2022 by itself does not tell whether user refers to price,
    # year, or mileage; so a hint is needed, for ,e.g., the word "miles" in the
    # sentence

    # " <mileage><>",
    "<units_mileage_mi><> <mileage><>",
    "<other><> <units_mileage_mi><> <mileage><>",
    "<units_mileage_mi><> <mileage><> <other><>",
    " <other><> <units_mileage_mi><> <mileage><> <other><>",
    "<mileage><> <units_mileage_mi><>",
    "<other><> <mileage><> <units_mileage_mi><>",
    "<mileage><> <units_mileage_mi><> <other><>",
    " <other><> <mileage><> <units_mileage_mi><> <other><>",

    # "  <mileage><> <range2><-> <mileage><> ",
    " <units_mileage_mi><> <mileage><>   <range2><> <mileage><> ",
    " <other><> <units_mileage_mi><> <mileage><>   <range2><> <mileage><> ",
    " <units_mileage_mi><> <mileage><>   <range2><> <mileage><> <other><>",
    " <other><> <units_mileage_mi><> <mileage><>   <range2><> <mileage><> <other><>",
    " <mileage><>   <range2><-> <mileage><>  <units_mileage_mi><> ",
    " <other><> <mileage><>   <range2><> <mileage><>  <units_mileage_mi><> ",
    " <mileage><>   <range2><> <mileage><>  <units_mileage_mi><> <other><>",
    " <other><> <mileage><>   <range2><> <mileage><>  <units_mileage_mi><> <other><>",

    # " <range1><>  <mileage><> <mileage><>  ",
    " <range1><>   <units_mileage_mi><> <mileage><> <mileage><>  ",
    " <other><> <range1><>   <units_mileage_mi><> <mileage><> <mileage><>  ",
    " <range1><>   <units_mileage_mi><> <mileage><> <mileage><>  <other><>",
    " <other><> <range1><>   <units_mileage_mi><> <mileage><> <mileage><>  <other><>",
    " <range1><>   <units_mileage_mi><> <mileage><> <other><and> <mileage><>  ",
    "<other><> <range1><>   <units_mileage_mi><> <mileage><> <other><and> <mileage><>  ",
    " <range1><>   <units_mileage_mi><> <mileage><>  <other><and>  <mileage><> <other><> ",
    " <other><> <range1><>   <units_mileage_mi><> <mileage><> <other><and> <mileage><>  <other><>",
    " <range1><>  <mileage><>  <units_mileage_mi><> <mileage><>  ",
    " <other><> <range1><>  <mileage><>  <units_mileage_mi><> <mileage><>  ",
    " <range1><>  <mileage><>  <units_mileage_mi><> <mileage><>  <other><>",
    " <other><> <range1><>  <mileage><>  <units_mileage_mi><> <mileage><>  <other><>",
    " <range1><>  <mileage><> <mileage><>   <units_mileage_mi><> ",
    " <other><> <range1><>  <mileage><> <mileage><>   <units_mileage_mi><> ",
    " <range1><>  <mileage><> <mileage><>   <units_mileage_mi><> <other><>",
    "<other><> <range1><>  <mileage><> <mileage><>   <units_mileage_mi><> <other><>",
    " <range1><>  <mileage><>  <other><and> <mileage><>   <units_mileage_mi><> ",
    " <other><> <range1><>  <mileage><>  <other><and> <mileage><>   <units_mileage_mi><> ",
    " <range1><>  <mileage><>  <other><and> <mileage><>   <units_mileage_mi><> <other><>",
    "<other><> <range1><>  <mileage><> <other><and>  <mileage><>   <units_mileage_mi><> <other><>",

    # " <less><>  <mileage><>  ",
    " <less><>   <units_mileage_mi><> <mileage><>  ",
    " <other><> <less><>   <units_mileage_mi><> <mileage><>  ",
    " <less><>   <units_mileage_mi><> <mileage><>   <other><> ",
    "  <other><> <less><>   <units_mileage_mi><> <mileage><>   <other><> ",
    " <less><> <other><than>  <units_mileage_mi><> <mileage><>  ",
    "  <other><> <less><> <other><than>  <units_mileage_mi><> <mileage><>  ",
    " <less><> <other><than>  <units_mileage_mi><> <mileage><>   <other><> ",
    "  <other><> <less><> <other><than>  <units_mileage_mi><> <mileage><>  <other><>  ",
    " <less><>  <mileage><>   <units_mileage_mi><> ",
    "  <other><> <less><>  <mileage><>   <units_mileage_mi><> ",
    " <less><>  <mileage><>   <units_mileage_mi><>  <other><> ",
    "  <other><> <less><>  <mileage><>   <units_mileage_mi><>  <other><> ",
    " <less><>  <other><than>   <mileage><>   <units_mileage_mi><> ",
    "  <other><> <less><>  <other><than>   <mileage><>   <units_mileage_mi><> ",
    " <less><>  <other><than>   <mileage><>   <units_mileage_mi><>  <other><> ",
    "  <other><> <less><>  <other><than>   <mileage><>   <units_mileage_mi><>  <other><> ",

    # "  <mileage><>  <less><> ",
    "  <units_mileage_mi><>  <mileage><>  <less><> ",
    "  <other><>  <units_mileage_mi><>  <mileage><>  <less><> ",
    "  <units_mileage_mi><>  <mileage><>  <less><>  <other><> ",
    "   <other><> <units_mileage_mi><>  <mileage><>  <less><>  <other><> ",
    "  <units_mileage_mi><>  <mileage><>  <other><or>  <less><> ",
    "  <other><>  <units_mileage_mi><>  <mileage><>  <other><or>  <less><> ",
    "  <units_mileage_mi><>  <mileage><>  <other><or>  <less><>  <other><> ",
    "   <other><> <units_mileage_mi><>  <mileage><>  <other><or>  <less><>  <other><> ",
    "  <mileage><>   <units_mileage_mi><> <less><> ",
    "   <other><> <mileage><>   <units_mileage_mi><> <less><> ",
    "  <mileage><>   <units_mileage_mi><> <less><>  <other><> ",
    "   <other><> <mileage><>   <units_mileage_mi><> <less><>  <other><> ",
    "  <mileage><>   <units_mileage_mi><> <other><or> <less><> ",
    "  <other><>  <mileage><>   <units_mileage_mi><> <other><or> <less><> ",
    "  <mileage><>   <units_mileage_mi><> <other><or> <less><>  <other><> ",
    "  <other><>  <mileage><>   <units_mileage_mi><> <other><or> <less><>  <other><> ",

    # " <more><>  <mileage><>  ",
    " <more><>   <units_mileage_mi><> <mileage><>  ",
    " <other><> <more><>   <units_mileage_mi><> <mileage><>  ",
    " <more><>   <units_mileage_mi><> <mileage><>   <other><> ",
    "  <other><> <more><>   <units_mileage_mi><> <mileage><>   <other><> ",
    " <more><> <other><than>  <units_mileage_mi><> <mileage><>  ",
    "  <other><> <more><> <other><than>  <units_mileage_mi><> <mileage><>  ",
    " <more><> <other><than>  <units_mileage_mi><> <mileage><>   <other><> ",
    "  <other><> <more><> <other><than>  <units_mileage_mi><> <mileage><>  <other><>  ",
    " <more><>  <mileage><>   <units_mileage_mi><> ",
    "  <other><> <more><>  <mileage><>   <units_mileage_mi><> ",
    " <more><>  <mileage><>   <units_mileage_mi><>  <other><> ",
    "  <other><> <more><>  <mileage><>   <units_mileage_mi><>  <other><> ",
    " <more><>  <other><than>   <mileage><>   <units_mileage_mi><> ",
    "  <other><> <more><>  <other><than>   <mileage><>   <units_mileage_mi><> ",
    " <more><>  <other><than>   <mileage><>   <units_mileage_mi><>  <other><> ",
    "  <other><> <more><>  <other><than>   <mileage><>   <units_mileage_mi><>  <other><> ",

    # "  <mileage><>  <>more<> ",
    "  <units_mileage_mi><>  <mileage><>  <more><> ",
    "  <other><>  <units_mileage_mi><>  <mileage><>  <more><> ",
    "  <units_mileage_mi><>  <mileage><>  <more><>  <other><> ",
    "   <other><> <units_mileage_mi><>  <mileage><>  <more><>  <other><> ",
    "  <units_mileage_mi><>  <mileage><>  <other><or>  <more><> ",
    "  <other><>  <units_mileage_mi><>  <mileage><>  <other><or>  <more><> ",
    "  <units_mileage_mi><>  <mileage><>  <other><or>  <more><>  <other><> ",
    "   <other><> <units_mileage_mi><>  <mileage><>  <other><or>  <more><>  <other><> ",
    "  <mileage><>   <units_mileage_mi><> <more><> ",
    "   <other><> <mileage><>   <units_mileage_mi><> <more><> ",
    "  <mileage><>   <units_mileage_mi><> <more><>  <other><> ",
    "   <other><> <mileage><>   <units_mileage_mi><> <more><>  <other><> ",
    "  <mileage><>   <units_mileage_mi><> <other><or> <more><> ",
    "  <other><>  <mileage><>   <units_mileage_mi><> <other><or> <more><> ",
    "  <mileage><>   <units_mileage_mi><> <other><or> <more><>  <other><> ",
    "  <other><>  <mileage><>   <units_mileage_mi><> <other><or> <more><>  <other><> ",
)
year_segments = (
    # a number, e.g. 2022 by itself does not tell whether user refers to price,
    # year, or mileage; so a hint is needed, for ,e.g., the word "year" in the
    # sentence. ***Actually I want a number between 1950 - 2023 without units
    # to be recognized by the model with a label of year
    # IMPORTANT: The following sentence does not have a hint of "year" but
    # the context contains brand_value/model_value which the model must learn:
    # "i want <year><2023> red toyota  ",

    # " <year><>",
    # " <year><>" must NOT appear between other-words because it is only
    # recognized when it is associated with brand/model/color; it is in
    # mixed_segments
    "  <other><year>  <year><> ",
    "   <other><> <other><year>  <year><> ",
    "  <other><year>  <year><>  <other><> ",
    " <other><> <other><year>  <year><>  <other><> ",
    " <year><>  <other><year>  ",
    " <other><> <year><>  <other><year>  ",
    " <year><>  <other><year>  <other><>  ",
    "  <other><> <year><>  <other><year>   <other><> ",

    # " <year><> <range2><> <year><>  ",
    # " <year><> <range2><> <year><>  " is in mixed_segments
    "  <other><year>  <year><> <range2><> <year><>  ",
    "  <other><>  <other><year>  <year><> <range2><> <year><>  ",
    "  <other><year>  <year><> <range2><> <year><>  <other><>  ",
    "   <other><> <other><year>  <year><> <range2><> <year><>   <other><> ",
    " <year><> <range2><> <year><>  <other><year>  ",
    "  <other><> <year><> <range2><> <year><>  <other><year>  ",
    " <year><> <range2><> <year><>  <other><year>  <other><>  ",
    "  <other><> <year><> <range2><> <year><>  <other><year>   <other><> ",

    # " <range1><>  <year><> <year><>  ",
    # " <range1><>  <year><> <year><>  ", it is not used otherwise it would be
    # in mixed_segments
    " <range1><>    <other><year>  <year><> <year><>  ",
    "  <other><> <range1><>    <other><year>  <year><> <year><>  ",
    " <range1><>    <other><year>  <year><> <year><>  <other><>  ",
    "  <other><> <range1><>    <other><year>  <year><> <year><>   <other><> ",
    " <range1><>  <year><>   <other><year>  <year><>  ",
    "  <other><> <range1><>  <year><>   <other><year>  <year><>  ",
    " <range1><>  <year><>   <other><year>  <year><>   <other><> ",
    "  <other><> <range1><>  <year><>   <other><year>  <year><>   <other><> ",
    " <range1><>  <year><> <year><>    <other><year>  ",
    "  <other><> <range1><>  <year><> <year><>    <other><year>  ",
    " <range1><>  <year><> <year><>    <other><year>  <other><>  ",
    "  <other><> <range1><>  <year><> <year><>    <other><year>   <other><> ",

    # " <less><>  <year><>  ",
    # " <less><>  <year><>  ", it is in mixed_segments
    " <less><>   <other><year>  <year><>  ",
    "  <other><> <less><>   <other><year>  <year><>  ",
    " <less><>   <other><year>  <year><>   <other><> ",
    "  <other><> <less><>   <other><year>  <year><>   <other><> ",
    " <less><>  <year><>   <other><year>  ",
    "  <other><> <less><>  <year><>   <other><year>  ",
    " <less><>  <year><>   <other><year>   <other><> ",
    "  <other><> <less><>  <year><>   <other><year>   <other><> ",
    " <less><>  <other><than>    <other><year>  <year><>  ",
    "  <other><> <less><>  <other><than>    <other><year>  <year><>  ",
    " <less><>  <other><than>    <other><year>  <year><>   <other><> ",
    "  <other><> <less><>  <other><than>    <other><year>  <year><>   <other><> ",

    # "  <year><>  <less><> ",
    # "  <year><>  <less><> ", it is in mixed_segments
    "   <other><year>   <year><>  <less><> ",
    "   <other><>  <other><year>   <year><>  <less><> ",
    "   <other><year>   <year><>  <less><>  <other><> ",
    "   <other><>  <other><year>   <year><>  <less><>  <other><> ",
    "  <year><>    <other><year>  <less><> ",
    "  <other><>  <year><>    <other><year>  <less><> ",
    "  <year><>    <other><year>  <less><>  <other><> ",
    "   <other><> <year><>    <other><year>  <less><>  <other><> ",
    "   <other><year>  <year><>    <other><or>   <less><> ",
    "   <other><>  <other><year>  <year><>    <other><or>   <less><> ",
    "   <other><year>  <year><>    <other><or>   <less><>  <other><> ",
    "   <other><>  <other><year>  <year><>    <other><or>   <less><>  <other><> ",

    # " <more><>  <year><>  ",
    # " <more><>  <year><>  ", it is in mixed_segments
    " <more><>   <other><year>  <year><>  ",
    "  <other><> <more><>   <other><year>  <year><>  ",
    " <more><>   <other><year>  <year><>   <other><> ",
    "  <other><> <more><>   <other><year>  <year><>   <other><> ",
    " <more><>  <year><>   <other><year>  ",
    "  <other><> <more><>  <year><>   <other><year>  ",
    " <more><>  <year><>   <other><year>   <other><> ",
    "  <other><> <more><>  <year><>   <other><year>   <other><> ",
    " <more><>  <other><than>    <other><year>  <year><>  ",
    "  <other><> <more><>  <other><than>    <other><year>  <year><>  ",
    " <more><>  <other><than>    <other><year>  <year><>   <other><> ",
    "  <other><> <more><>  <other><than>    <other><year>  <year><>   <other><> ",

    # "  <year><>  <more><> ",
    # "  <year><>  <more><> ", it is in mixed_segments
    "   <other><year>   <year><>  <more><> ",
    "   <other><>  <other><year>   <year><>  <more><> ",
    "   <other><year>   <year><>  <more><>  <other><> ",
    "   <other><>  <other><year>   <year><>  <more><>  <other><> ",
    "  <year><>    <other><year>  <more><> ",
    "  <other><>  <year><>    <other><year>  <more><> ",
    "  <year><>    <other><year>  <more><>  <other><> ",
    "   <other><> <year><>    <other><year>  <more><>  <other><> ",
    "   <other><year>  <year><>    <other><or>   <more><> ",
    "   <other><>  <other><year>  <year><>    <other><or>   <more><> ",
    "   <other><year>  <year><>    <other><or>   <more><>  <other><> ",
    "   <other><>  <other><year>  <year><>    <other><or>   <more><>  <other><> ",
)
remove_segments = (
    "<remove><> <restore><> <remove><> ",
    "<remove><> <brand><>  <model><>  <color><> ",
    "<remove><> <units_price_$><> <price><> <units_mileage_mi><> <mileage><> <year><> ",
    "<restore><>  <setting><> <remove><>  <color><>  <year><> ",
    "<restore><>  <setting><> <remove><>  <brand><>  <year><> ",
)
mixed_segments = (
    "<units_price_$><$> <price><>  <range2><-> <price><> <mileage><> <range2><-> <mileage><> <units_mileage_mi><> ",
    "<units_price_$><$> <price><>  <range2><-> <price><> <less><>  <other><than>  <mileage><> <units_mileage_mi><> ",
    "<units_price_$><$> <price><>  <range2><-> <price><> <mileage><> <units_mileage_mi><>  <other><or>  <less><>",
    "<less><> <units_price_$><> <price><> <mileage><> <range2><-> <mileage><> <units_mileage_mi><> ",
    "<less><> <units_price_$><> <price><> <less><>  <other><than>  <mileage><> <units_mileage_mi><>",
    "<less><> <units_price_$><> <price><> <mileage><> <units_mileage_mi><>  <other><or>  <less><>",
    "<mileage><> <range2><-> <mileage><> <units_mileage_mi><> <units_price_$><> <price><> <range2><-> <price><> ",
    "<mileage><> <range2><-> <mileage><> <units_mileage_mi><> <less><> <units_price_$><> <price><>",
    "<mileage><> <range2><-> <mileage><> <units_mileage_mi><> <price><> <units_price_$><>  <other><or>  <less><>",
    "<less><> <mileage><> <units_mileage_mi><> <units_price_$><> <price><> <range2><-> <price><> ",
    "<less><> <mileage><> <units_mileage_mi><> <less><> <units_price_$><> <price><>",
    "<less><> <mileage><> <units_mileage_mi><> <price><> <units_price_$><>  <other><or>  <less><>",
    "<range1><> <units_price_$><> <price><> <price><> <range1><> <mileage><> <mileage><> <units_mileage_mi><>",
    "<range1><> <units_price_$><> <price><> <price><> <less><> <mileage><> <units_mileage_mi><>",
    "<range1><> <units_price_$><> <price><> <price><> <mileage><> <units_mileage_mi><> <less><>",
    "<less><> <units_price_$><> <price><> <range1><> <mileage><> <mileage><> <units_mileage_mi><>",
    "<range1><> <mileage><> <mileage><> <units_mileage_mi><> <range1><> <units_price_$><> <price><> <price><>",
    "<range1><> <mileage><> <mileage><> <units_mileage_mi><> <less><> <units_price_$><> <price><>",
    "<range1><> <mileage><> <mileage><> <units_mileage_mi><> <price><> <units_price_$><> <less><>",
    "<less><> <mileage><> <units_mileage_mi><> <range1><> <units_price_$><> <price><> <price><>",
    "<year><> <brand><> ",
    "<year><> <model><> ",
    "<year><> <color><> ",
    " <year><> <range2><> <year><>  <brand><>  ",
    " <year><> <range2><> <year><>  <model><>  ",
    " <year><> <range2><> <year><>  <color><>  ",

    # following is how the user should normally type
    " <color><> <year><> <brand><> <model><>  <mileage><>   <range2><-> <mileage><>  <units_mileage_mi><>   <range1><>   <units_price_$><> <price><>  <other><and>  <price><>  ",
    " <model><>  <year><> <range2><> <year><>   <color><>    <less><>  <other><than>   <price><> <units_price_$><> and  <less><>  <other><than>  <mileage><>   <units_mileage_mi><> ",
    " <color><>  <more><>   <other><than>  <year><>  <model><> <brand><>   <units_price_$><> <price><>   <range2><-> <price><>  <mileage><>   <range2><->  <units_mileage_mi><mi> <mileage><>  ",
    "<brand><> <less><>  <year><>   <color><> <model><>   <range1><>  <mileage><>  <units_mileage_mi><>  <other><and>  <mileage><>    <range1><>   <price><> <price><>  <units_price_$><> ",
    "<mileage><>   <range2><-> <mileage><>  <units_mileage_mi><>   <range1><>   <units_price_$><> <price><>   <other><and>   <price><>  <year><> <color><> <brand><> <model><>  ",
    "  <year><> <range2><> <year><>   <color><>    <less><>  <other><than>   <units_price_$><> <price><> <model><>   <less><>  <other><than>   <mileage><>   <units_mileage_mi><> ",
    " <color><>  <more><>   <other><than>  <year><>  <model><>  <units_price_$><> <price><>   <range2><-> <price><>  <mileage><>   <range2><->  <units_mileage_mi><mi> <mileage><>   <brand><> ",
    "<brand><>  <color><> <model><>   <range1><>  <mileage><>    <other><and>   <mileage><>  <units_mileage_mi><>   <range1><>   <units_price_$><> <price><> <price><>   <less><>  <year><> ",
    " <less><>  <other><than>   <price><> <units_price_$><>   <other><and>    <mileage><>   <units_mileage_mi><> ",
    "<units_price_$><$> <price><> <mileage><> <units_mileage_mi><> ",
    "<year><> <color><> <brand><> <model><> <less><>  <other><than>   <units_price_$><> <price><>   <other><and>    <less><>  <other><than>   <mileage><>   <units_mileage_mi><> ",
    " <color><>  <more><>   <other><than>  <year><>  <model><> <brand><>  <range1><>  <mileage><>  <units_mileage_mi><>   <other><and>   <mileage><>    <range1><>   <units_price_$><> <price><> <price><>  ",
    "<brand><> <less><>  <year><>   <color><> <model><>   <units_price_$><> <price><>   <range2><-> <price><>  <mileage><>   <range2><->  <units_mileage_mi><mi> <mileage><> ",
    " <model><>  <year><> <range2><> <year><>   <color><>      <mileage><>   <range2><-> <mileage><>  <units_mileage_mi><>   <range1><>   <units_price_$><> <price><>   <other><and>   <price><> ",
    "<units_price_$><$> <price><> <mileage><> <units_mileage_mi><> ",

    # NN must sometimes label entityWrds as "O"; usually (but not always) when they are followed by an "O" word
    "<color><>  <other><through>   <other><year> <brand><>  ",  # In Predict, NN must not label "through" as range2 label
    "   <mileage><>   <range2><-> <mileage><>  <units_mileage_mi><>  <other><>   <other><higher>   <other><>   <other><lower>   <other><>   <units_price_$><> <price><>  <range2><-> <price><>  <other><>   <other><little>   <other><> ",
    "<color><>  <other><smaller>  <brand><>  <price><>  <range2><-> <price><> <units_price_$><> ",
    " <other><>   <other><more>   <brand><>   <other><above>  <model><>  <other><>   <other><through>   <other><> <less><>   <other><than>  <units_price_$><> <price><>",
    "  <other><>  <other><under>   <other><>  <model><>   <other><over>  <brand><>  <other><range>   <color><>  <larger><>  <other><than>   <mileage><>   <units_mileage_mi><> ",
    " <other><>   <other><mileage>   <other><>   <other><$>   <other><>   <other><dollar>    <other><between>    <other><price>   <other><>  <other><color>  <other><>  <price><> <units_price_$><>",
    " <other><>  <other><to>   <other><mileages>   <other><>   <other><dollars>   <other><above>    <other><>   <other><below>   <other><> <units_price_$><> <price><>   <range2><-> <price><>  <other><>   <other><greater>   <other><>   <other><little>   <other><>  <mileage><>   <units_mileage_mi><>",
    "  <other><remove>  <other><> <color><>  <other><mi> <brand><>",
    " <model><> <other><delete>  <other><> <model><>  <other><mile> <model><>",
    "<units_price_$><> <price><>  <range2><-> <price><>  <other><clear>  <other><miles>  <other><> ",
    "  <other><brand>  <other><model>  <other><color>  <other><make> ",

    # NN must not always label 1970 - 2024 as belonging to year
    "<color><> <other><__year>  <other><>  <price><__year> <range2><> <price><__year> <units_price_$><>   <less><less>  <other><than>  <mileage><__year> <units_mileage_mi><miles> ",
    " <mileage><__year> <units_mileage_mi><>   <price><__year> <units_price_$><> <other><__year> ",
    " <other><__year>  <other><> <units_price_$><> <price><__year> <more><> <mileage><__year> <units_mileage_mi><>",
    "  <units_price_$><> <price><__year> <mileage><__year> <units_mileage_mi><miles> <more><> ",
)
specific_segments = (
    "<multilabel><> ",
    "<assoc_brand_modelNum><> ",
)
"""
style_segments = (
    "<style><>",
    "style <style><convertible>  ",
    " <style><convertible> style ",
    "style <style><sedan>  ",
    " <style><sedan> style ",
    "style <style><pickup>  ",
    " <style><pickup> style ",
    "style <style><wagon>  ",
    " <style><wagon> style ",
    "style <style><van>  ",
    " <style><van> style ",
    "style <style><coupe>  ",
    " <style><coupe> style ",
    "style <style><suv>  ",
    " <style><suv> style ",
)
"""
# "setting_segments" are in "remove segment" because restore in the middle of a
# full sentence auto-removes everything before it
# setting_segments = (
#     "<restore><> <setting><>",
#     "<restore><>",
# )
segments_weights = {
    "brand": {
        "segments": brand_segments,
        "weight": 50
        # weights are relative to each other
        # brand_probability = brand_weight / (sum of all weights)
    },
    "model": {
        "segments": model_segments,
        "weight": 50
    },
    "color": {
        "segments": color_segments,
        "weight": 50
    },
    "price": {
        "segments": price_segments,
        "weight": 50
    },
    "year": {
        "segments": year_segments,
        "weight": 50
    },
    "mileage": {
        "segments": mileage_segments,
        "weight": 50
    },
    "remove": {
        "segments": remove_segments,
        "weight": 10
    },
    "mixed": {
        "segments": mixed_segments,
        "weight": 50
    },
    "specific": {
        "segments": specific_segments,
        "weight": 10
    },
    # "style": style_segments,
    # "setting": setting_segments,
}

cmdsLbls = {
    "restore": (
        "restore",
        "retrieve",
        "retreive",
    ),
    "more": ("over", "more", "higher", "above", "larger", "greater"),
    "less": ("under", "less", "lower", "below", "smaller", "little"),
    "range1": ("between", "range"),  # e.g. between $5000 and $7000
    "range2": ("-", "to", "through"),  # e.g. $5000 - $7000
    "remove": ("remove", "delete", "clear"),
}
# carEntityNum must precede following commands
cmdsLbls_follow_carEntityNum = ("range2", )

unitsLbls = {
    "units_price_$": ("$", "dollar", "dollars"),
    "units_mileage_mi": (
        "mi",
        "mile",
        "miles",
        "mileage",
        "mileages",
    ),
}
carEntityNumLbl2unitLbl = {
    "price": "units_price_$",
    "mileage": "units_mileage_mi",
    "year": None,
}
carEntityNumLbls_requireUnit = (
    "price",
    "mileage",
)
units = ()
for v in unitsLbls.values():
    units += v

hyphen = "-"

carEntityNonNumLbls = (
    "brand",
    "model",
    "color",
    # "style",    # needed in Fill_entityWrds.py and deleted
)
synonyms_for_carEntityNonNumLbls = {
    # different datasets use different names, e.g. "make" instead of "brand"
    "brand": ["brand", "make"],
    "model": ["model"],
    "color": ["color"],
    # "style": ["body_style", "body_styles"],
    # 'style' needed in Fill_entityWrds.py and deleted
}

carEntityNumLbls = (
    "mileage",
    "price",
    "year",
)

carEntityLbls = (carEntityNonNumLbls + carEntityNumLbls)


def mileage_func() -> str:
    if random.getrandbits(1):
        return str(round(random.uniform(0, 9999999999), 2))
    else:
        return str(int(random.uniform(0, 9999999999)))


def price_func() -> str:
    if random.getrandbits(1):
        return str(round(random.uniform(0, 9999999999), 2))
    else:
        return str(int(random.uniform(0, 9999999999)))


def setting_func() -> str:
    return str(random.randint(0, 9999999999))
    #return str(random.randint(1, 5))


def year_func() -> str:  # this generator has infinite loop
    year: int = 2024
    all_years_done: bool = False
    while True:
        if year == 1970:
            all_years_done = True
            year = 2024
        yield str(year), all_years_done
        year -= 1


entityLbls_for_numEntityWrds_mapTo_func = {
    "mileage": mileage_func,
    "price": price_func,
    "setting": setting_func,
}
entityLbls_for_numEntityWrds_mapTo_genFunc = {
    "year": year_func,
}
