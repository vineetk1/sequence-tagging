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
(i) If a string already has single-quotes then it should be represented with
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
     entityLbl = "color"; entityWrd (aka keyword) = "red"

***NOTE: that <> means that code will fill this placeholder with a string
            or integer based on the context

Formats follow:
---------------
(1) <entityLbl><>
Code picks entityWrd belonging to entityLbl

(2) <entityLbl><entityWrd>

Formats of <entityWrd>
---------------------------
** <entityLbl><>
Code picks an entityWrd belonging to entityLbl

** <price><> vs. <price><___intFloat> => No difference

** <mileage><> vs. <mileage><___intFloat> => No difference

** (i) <year><> vs. (ii) <entityLbl><___rdmYear> vs. (iii) <other><___notYear>
(i) everytime used, entityWrd is the next integer in the range of Year
(ii) entityWrd is a random integer between the range of Year
(iii) entityWrd is a random integer that is NOT between the range of Year

entityWrd starts with three underscores, ___unique_name:
    Following are used:
        <other><___entWrdLblOther>
        <other><___rdmYear>  <price><___rdmYear>    <year><___rdmYear>   <other><___notYear>
    Following are not yet used:
        <price><___intFloat>

Some notes:
-----------
+ The same numbers are used for all the numEntities. For example a number
    of 2004 is used in all the following cases: (1) year 2004, (2) $2004,
    (3) 2004 miles, (4) "setting 2004" is NOT used because only numbers
    between 1 through 20 are valid
+ Instead of using a -ve, a previous range can be removed by
    mentioning a new range

NN must learn when to label or not-label the entityWrds:
--------------------------------------------------------
Add to segments certain entityWrds (e.g. between, to, through) that the NN must learn to NOT label. Example:
"i have to tell you that <year><> <range2><> <year><>   <mileage><>   <range2><-> <mileage><>  <units_mileage_mi><>  <units_price_$><> <price><>   <range2><-> <price><>  is between a few cars i want through"
In the above sentence, the NN must NOT label the "to", "between", "through" {i think that "to", "between", "through" should be randomly generated from some list containing such words}


Segments that are NOT ALLOWED: *****MAJOR PROBLEM************
------------------------------
" <price><>   <units_price_$><>  <range2><-> <price><> ",
" <price><>   <range2><->  <units_price_$><> <price><> ",
Reason: if "units_price_$" is "dollar" then its typo could become "dolkar". The
function Utilities.py: userIn_filter_splitWords(userIn) removes the hyphen
because "dolkar" is neither a number or a unit
UPDATE: I think the above two segments MUST be allowed. If a user misspells
"dollar" then the code will remove the hyphen and the output will be different
than expected; the user made the mistake and in this case the NN was not smart
enough to correct the spelling of "dolkar"
"""
import random

PLACEHOLDER_ID_START = "<"
PLACEHOLDER_ID_END = ">"

# Following sentences are different from segments; they are used only once in
# the dataset
train_sentences = ()
val_sentences = (
    # following is how the user should normally type

    # NN must sometimes label entityWrds as "O"; usually (but not always) when
    # they are followed by an "O" word
    " <other><___entWrdLblOther> <color><>  <other><___entWrdLblOther> <TBD><___assoc_brand_modelNum>   <other><___entWrdLblOther>  <other><___entWrdLblOther>  <other><> <less><>  <other><than>   <units_price_$><> <price><> <TBD><___multilabel> ",
    " <other><___entWrdLblOther>  <other><___entWrdLblOther>   <other><>  <mileage><>   <range2><-> <mileage><>  <units_mileage_mi><>  <other><___entWrdLblOther>  <other><>  <year><> <range2><> <year><>  <other><___entWrdLblOther>  <other><> <units_price_$><> <price><>   <range2><-> <price><>  <other><___entWrdLblOther> ",
    " <other><___entWrdLblOther>  <other><>  <other><___rdmYear>  <other><___entWrdLblOther>  <other><>  <price><> <units_price_$><> <color><>   <other><___entWrdLblOther>   <brand><> ",
    "  <brand><>  <other><___entWrdLblOther>  <other><> <less><>  <other><than>  <units_price_$><>   <price><>  <other><___entWrdLblOther>  <other><___entWrdLblOther>   ",
    "  <other><___entWrdLblOther>  <model><>   <other><___entWrdLblOther>  <other><>  <price><>   <range2><-> <price><>  <units_price_$><>   <other><___entWrdLblOther>  <other><>  <mileage><>   <units_mileage_mi><> ",
    "<price><> <units_price_$><>  <other><___entWrdLblOther>  <brand><>  ",
    "  <other><___entWrdLblOther> <price><>   <range2><-> <price><> <units_price_$><> <other><___entWrdLblOther>  <other><___entWrdLblOther> <color><>  <other><___entWrdLblOther>  ",
)
test_sentences = (
    # following is how the user should normally type

    # NN must sometimes label entityWrds as "O"; usually (but not always) when they are followed by an "O" word
    " <other><___entWrdLblOther> <other><___entWrdLblOther> ",
    " <other><___entWrdLblOther> <brand><>  <other><___entWrdLblOther>  <brand><>  ",
    " <other><___entWrdLblOther> <other><___entWrdLblOther>   <other><>  <range2><-> <mileage><>  <units_mileage_mi><>  <units_price_$><> <price><>   <other><___entWrdLblOther>  <other><>  <other><___entWrdLblOther>  <model><> ",
    "<other><>  <other><>  <other><>   <other><___rdmYear>  <other><>   <color><>  <other><> <brand><> ",
    " <other><>   <brand><>  <brand><>   <other><___entWrdLblOther>   <other><>   <units_price_$><> <price><>",
    "   <year><> <model><>  <other><___entWrdLblOther>     <brand><>  <other><> <other><___entWrdLblOther>     <color><>  <less><>  <other><>  <mileage><>   <units_mileage_mi><> ",
    " <year><> <TBD><___multilabel>  <other><___entWrdLblOther>  <TBD><___assoc_brand_modelNum> <price><>   <range2><-> <price><> <units_price_$><>",
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
    "<other><___entWrdLblOther>   <brand><>  ",
    "<brand><>  <other><> ",
    "<other><> <brand><>  <other><> ",
    # why not include: <other><brand> <brand><>
)
model_segments = (
    # neural-net must memorize, e.g., camry is model, without any hints
    "<model><>  ",
    " <other><___entWrdLblOther>   <model><>  ",
    "<model><>  <other><> ",
    "<other><> <model><>  <other><> ",
    # why not include: <other><model> <model><>
)
color_segments = (
    # neural-net must memorize, e.g., red is color, without any hints
    "<color><>  ",
    "<other><> <color><>  ",
    "<color><>  <other><> ",
    " <other><___entWrdLblOther>   <color><>  <other><> ",
    # why not include: <other><color> <color><>
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
    "<other><> <price><> <units_price_$><dollar>",
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
    " <range1><>   <units_price_$><> <price><> <other><> <price><>  ",
    "<other><> <range1><>   <units_price_$><> <price><> <other><> <price><>  ",
    " <range1><>   <units_price_$><> <price><>  <other><>  <price><> <other><> ",
    " <other><> <range1><>   <units_price_$><> <price><> <other><> <price><>  <other><>",
    " <range1><>  <price><>  <units_price_$><> <price><>  ",
    " <other><> <range1><>  <price><>  <units_price_$><> <price><>  ",
    " <range1><>  <price><>  <units_price_$><> <price><>  <other><>",
    " <other><> <range1><>  <price><>  <units_price_$><> <price><>  <other><>",
    " <range1><>  <price><> <price><>   <units_price_$><> ",
    " <other><> <range1><>  <price><> <price><>   <units_price_$><> ",
    " <range1><>  <price><> <price><>   <units_price_$><> <other><>",
    "<other><> <range1><>  <price><> <price><>   <units_price_$><> <other><>",
    " <range1><>  <price><>  <other><> <price><>   <units_price_$><> ",
    " <other><> <range1><>  <price><>  <other><> <price><>   <units_price_$><> ",
    " <range1><>  <price><>  <other><> <price><>   <units_price_$><> <other><>",
    "<other><> <range1><>  <price><> <other><>  <price><>   <units_price_$><> <other><>",

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
    "  <units_price_$><>  <price><>  <other><>  <less><> ",
    "  <other><>  <units_price_$><>  <price><>  <other><>  <less><> ",
    "  <units_price_$><>  <price><>  <other><>  <less><>  <other><> ",
    "   <other><> <units_price_$><>  <price><>  <other><>  <less><>  <other><> ",
    "  <price><>   <units_price_$><> <less><> ",
    "   <other><> <price><>   <units_price_$><> <less><> ",
    "  <price><>   <units_price_$><> <less><>  <other><> ",
    "   <other><> <price><>   <units_price_$><> <less><>  <other><> ",
    "  <price><>   <units_price_$><> <other><> <less><> ",
    "  <other><>  <price><>   <units_price_$><> <other><> <less><> ",
    "  <price><>   <units_price_$><> <other><> <less><>  <other><> ",
    "  <other><>  <price><>   <units_price_$><> <other><> <less><>  <other><> ",

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
    "  <units_price_$><>  <price><>  <other><>  <more><> ",
    "  <other><>  <units_price_$><>  <price><>  <other><>  <more><> ",
    "  <units_price_$><>  <price><>  <other><>  <more><>  <other><> ",
    "   <other><> <units_price_$><>  <price><>  <other><>  <more><>  <other><> ",
    "  <price><>   <units_price_$><> <more><> ",
    "   <other><> <price><>   <units_price_$><> <more><> ",
    "  <price><>   <units_price_$><> <more><>  <other><> ",
    "   <other><> <price><>   <units_price_$><> <more><>  <other><> ",
    "  <price><>   <units_price_$><> <other><> <more><> ",
    "  <other><>  <price><>   <units_price_$><> <other><> <more><> ",
    "  <price><>   <units_price_$><> <other><> <more><>  <other><> ",
    "  <other><>  <price><>   <units_price_$><> <other><> <more><>  <other><> ",
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
    " <range1><>   <units_mileage_mi><> <mileage><> <other><> <mileage><>  ",
    "<other><> <range1><>   <units_mileage_mi><> <mileage><> <other><> <mileage><>  ",
    " <range1><>   <units_mileage_mi><> <mileage><>  <other><>  <mileage><> <other><> ",
    " <other><> <range1><>   <units_mileage_mi><> <mileage><> <other><> <mileage><>  <other><>",
    " <range1><>  <mileage><>  <units_mileage_mi><> <mileage><>  ",
    " <other><> <range1><>  <mileage><>  <units_mileage_mi><> <mileage><>  ",
    " <range1><>  <mileage><>  <units_mileage_mi><> <mileage><>  <other><>",
    " <other><> <range1><>  <mileage><>  <units_mileage_mi><> <mileage><>  <other><>",
    " <range1><>  <mileage><> <mileage><>   <units_mileage_mi><> ",
    " <other><> <range1><>  <mileage><> <mileage><>   <units_mileage_mi><> ",
    " <range1><>  <mileage><> <mileage><>   <units_mileage_mi><> <other><>",
    "<other><> <range1><>  <mileage><> <mileage><>   <units_mileage_mi><> <other><>",
    " <range1><>  <mileage><>  <other><> <mileage><>   <units_mileage_mi><> ",
    " <other><> <range1><>  <mileage><>  <other><> <mileage><>   <units_mileage_mi><> ",
    " <range1><>  <mileage><>  <other><> <mileage><>   <units_mileage_mi><> <other><>",
    "<other><> <range1><>  <mileage><> <other><>  <mileage><>   <units_mileage_mi><> <other><>",

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
    "  <units_mileage_mi><>  <mileage><>  <other><>  <less><> ",
    "  <other><>  <units_mileage_mi><>  <mileage><>  <other><>  <less><> ",
    "  <units_mileage_mi><>  <mileage><>  <other><>  <less><>  <other><> ",
    "   <other><> <units_mileage_mi><>  <mileage><>  <other><>  <less><>  <other><> ",
    "  <mileage><>   <units_mileage_mi><> <less><> ",
    "   <other><> <mileage><>   <units_mileage_mi><> <less><> ",
    "  <mileage><>   <units_mileage_mi><> <less><>  <other><> ",
    "   <other><> <mileage><>   <units_mileage_mi><> <less><>  <other><> ",
    "  <mileage><>   <units_mileage_mi><> <other><> <less><> ",
    "  <other><>  <mileage><>   <units_mileage_mi><> <other><> <less><> ",
    "  <mileage><>   <units_mileage_mi><> <other><> <less><>  <other><> ",
    "  <other><>  <mileage><>   <units_mileage_mi><> <other><> <less><>  <other><> ",

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
    "  <units_mileage_mi><>  <mileage><>  <other><>  <more><> ",
    "  <other><>  <units_mileage_mi><>  <mileage><>  <other><>  <more><> ",
    "  <units_mileage_mi><>  <mileage><>  <other><>  <more><>  <other><> ",
    "   <other><> <units_mileage_mi><>  <mileage><>  <other><>  <more><>  <other><> ",
    "  <mileage><>   <units_mileage_mi><> <more><> ",
    "   <other><> <mileage><>   <units_mileage_mi><> <more><> ",
    "  <mileage><>   <units_mileage_mi><> <more><>  <other><> ",
    "   <other><> <mileage><>   <units_mileage_mi><> <more><>  <other><> ",
    "  <mileage><>   <units_mileage_mi><> <other><> <more><> ",
    "  <other><>  <mileage><>   <units_mileage_mi><> <other><> <more><> ",
    "  <mileage><>   <units_mileage_mi><> <other><> <more><>  <other><> ",
    "  <other><>  <mileage><>   <units_mileage_mi><> <other><> <more><>  <other><> ",
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
    "   <other><year>  <year><>    <other><>   <less><> ",
    "   <other><>  <other><year>  <year><>    <other><>   <less><> ",
    "   <other><year>  <year><>    <other><>   <less><>  <other><> ",
    "   <other><>  <other><year>  <year><>    <other><>   <less><>  <other><> ",

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
    "   <other><year>  <year><>    <other><>   <more><> ",
    "   <other><>  <other><year>  <year><>    <other><>   <more><> ",
    "   <other><year>  <year><>    <other><>   <more><>  <other><> ",
    "   <other><>  <other><year>  <year><>    <other><>   <more><>  <other><> ",

    # "  <year><>  <brand/model/color><> ",
    " <year><>  <brand><> ",
    " <year><>  <model><> ",
    " <year><>  <color><> ",
    " <year><> <range2><> <year><>  <brand><>  ",
    " <year><> <range2><> <year><>  <model><>  ",
    " <year><> <range2><> <year><>  <color><>  ",
    " <year><>  <TBD><___assoc_brand_modelNum>  ",
    " <year><> <range2><> <year><>  <TBD><___assoc_brand_modelNum>   ",

    # "-ve ex: <other><___rdmYear>  (not <brand/model/color><> ",
    "   <other><___rdmYear>   <other><> ",

    # "-ve ex: <other><___notYear> <brand/model/color><> "
    "   <other><___notYear>   <other><> ",
    "   <other><___notYear>   <brand><> ",
    "   <other><___notYear>   <model><> ",
    "   <other><___notYear>   <color><> ",
)
remove_restore_segments = (
    "<remove><> <restore><> <remove><> ",
    "<remove><> <brand><>  <model><>  <color><> ",
    "<remove><> <units_price_$><> <price><> <units_mileage_mi><> <mileage><> <year><> ",
    "<restore><>  <setting><> <remove><>  <color><>  <year><> ",
    "<restore><>  <setting><> <remove><>  <brand><>  <year><> ",
)
mixed_segments = (
    "<units_price_$><$> <price><>  <range2><-> <price><> <mileage><> <range2><-> <mileage><> <units_mileage_mi><> ",
    "<units_price_$><$> <price><>  <range2><-> <price><> <less><>  <other><than>  <mileage><> <units_mileage_mi><> ",
    "<units_price_$><$> <price><>  <range2><-> <price><> <mileage><> <units_mileage_mi><>  <other><>  <less><>",
    "<less><> <units_price_$><> <price><> <mileage><> <range2><-> <mileage><> <units_mileage_mi><> ",
    "<less><> <units_price_$><> <price><> <less><>  <other><than>  <mileage><> <units_mileage_mi><>",
    "<less><> <units_price_$><> <price><> <mileage><> <units_mileage_mi><>  <other><>  <less><>",
    "<mileage><> <range2><-> <mileage><> <units_mileage_mi><> <units_price_$><> <price><> <range2><-> <price><> ",
    "<mileage><> <range2><-> <mileage><> <units_mileage_mi><> <less><> <units_price_$><> <price><>",
    "<mileage><> <range2><-> <mileage><> <units_mileage_mi><> <price><> <units_price_$><>  <other><>  <less><>",
    "<less><> <mileage><> <units_mileage_mi><> <units_price_$><> <price><> <range2><-> <price><> ",
    "<less><> <mileage><> <units_mileage_mi><> <less><> <units_price_$><> <price><>",
    "<less><> <mileage><> <units_mileage_mi><> <price><> <units_price_$><>  <other><>  <less><>",
    "<range1><> <units_price_$><> <price><> <price><> <range1><> <mileage><> <mileage><> <units_mileage_mi><>",
    "<range1><> <units_price_$><> <price><> <price><> <less><> <mileage><> <units_mileage_mi><>",
    "<range1><> <units_price_$><> <price><> <price><> <mileage><> <units_mileage_mi><> <less><>",
    "<less><> <units_price_$><> <price><> <range1><> <mileage><> <mileage><> <units_mileage_mi><>",
    "<range1><> <mileage><> <mileage><> <units_mileage_mi><> <range1><> <units_price_$><> <price><> <price><>",
    "<range1><> <mileage><> <mileage><> <units_mileage_mi><> <less><> <units_price_$><> <price><>",
    "<range1><> <mileage><> <mileage><> <units_mileage_mi><> <price><> <units_price_$><> <less><>",
    "<less><> <mileage><> <units_mileage_mi><> <range1><> <units_price_$><> <price><> <price><>",

    # following is how the user should normally type
    " <color><> <year><> <brand><> <model><>  <mileage><>   <range2><-> <mileage><>  <units_mileage_mi><>   <range1><>   <units_price_$><> <price><>  <other><>  <price><>  ",
    " <model><>  <year><> <range2><> <year><>   <color><>    <less><>  <other><than>   <price><> <units_price_$><>  <less><>  <other><than>  <mileage><>   <units_mileage_mi><> ",
    " <color><>  <more><>   <other><than>  <year><>  <model><> <brand><>   <units_price_$><> <price><>   <range2><-> <price><>  <mileage><>   <range2><->  <units_mileage_mi><> <mileage><>  ",
    "<brand><> <less><>  <year><>   <color><> <model><>   <range1><>  <mileage><>  <units_mileage_mi><>  <other><>  <mileage><>    <range1><>   <price><> <price><>  <units_price_$><> ",
    "<mileage><>   <range2><-> <mileage><>  <units_mileage_mi><>   <range1><>   <units_price_$><> <price><>   <other><>   <price><>  <year><> <color><> <brand><> <model><>  ",
    "  <year><> <range2><> <year><>   <color><>    <less><>  <other><than>   <units_price_$><> <price><> <model><>   <less><>  <other><than>   <mileage><>   <units_mileage_mi><> ",
    " <color><>  <more><>   <other><than>  <year><>  <model><>  <units_price_$><> <price><>   <range2><-> <price><>  <mileage><>   <range2><->  <units_mileage_mi><> <mileage><>   <brand><> ",
    "<brand><>  <color><> <model><>   <range1><>  <mileage><>    <other><>   <mileage><>  <units_mileage_mi><>   <range1><>   <units_price_$><> <price><> <price><>   <less><>  <year><> ",
    " <less><>  <other><than>   <price><> <units_price_$><>   <other><>    <mileage><>   <units_mileage_mi><> ",
    "<units_price_$><$> <price><> <mileage><> <units_mileage_mi><> ",
    "<year><> <color><> <brand><> <model><> <less><>  <other><than>   <units_price_$><> <price><>   <other><>    <less><>  <other><than>   <mileage><>   <units_mileage_mi><> ",
    " <color><>  <more><>   <other><than>  <year><>  <model><> <brand><>  <range1><>  <mileage><>  <units_mileage_mi><>   <other><>   <mileage><>    <range1><>   <units_price_$><> <price><> <price><>  ",
    "<brand><> <less><>  <year><>   <color><> <model><>   <units_price_$><> <price><>   <range2><-> <price><>  <mileage><>   <range2><->  <units_mileage_mi><> <mileage><> ",
    " <model><>  <year><> <range2><> <year><>   <color><>      <mileage><>   <range2><-> <mileage><>  <units_mileage_mi><>   <range1><>   <units_price_$><> <price><>   <other><>   <price><> ",
    "<units_price_$><$> <price><> <mileage><> <units_mileage_mi><> ",

    # NN must sometimes label entityWrds as "O"; usually (but not always) when they are followed by an "O" word
    "<color><>  <other><___entWrdLblOther>   <other><___entWrdLblOther> <brand><>  ",  # In Predict, NN must not label "through" as range2 label
    "   <mileage><>   <range2><-> <mileage><>  <units_mileage_mi><>  <other><>   <other><___entWrdLblOther>   <other><>   <other><___entWrdLblOther>   <other><>   <units_price_$><> <price><>  <range2><-> <price><>  <other><>   <other><___entWrdLblOther>   <other><> ",
    "<color><>  <other><___entWrdLblOther>  <brand><>  <price><>  <range2><-> <price><> <units_price_$><> ",
    " <other><>  <TBD><___multilabel>  <other><___entWrdLblOther>   <brand><>   <other><___entWrdLblOther>  <model><>  <other><>   <other><___entWrdLblOther>   <other><> <less><>   <other><than>  <units_price_$><> <price><>",
    "  <TBD><___multilabel> <other><>  <TBD><___assoc_brand_modelNum> <other><___entWrdLblOther>   <other><>  <model><>   <other><___entWrdLblOther>  <brand><>  <other><___entWrdLblOther>   <color><>  <other><than>   <mileage><>   <units_mileage_mi><> ",
    " <other><>   <other><___entWrdLblOther>   <other><>   <other><___entWrdLblOther>   <other><>   <other><___entWrdLblOther>    <other><___entWrdLblOther>    <other><___entWrdLblOther>   <other><>  <other><___entWrdLblOther>  <other><>  <price><> <units_price_$><>",
    " <other><>  <other><___entWrdLblOther>   <other><___entWrdLblOther>   <other><>   <other><___entWrdLblOther>   <other><___entWrdLblOther>    <other><>   <other><___entWrdLblOther>   <other><> <units_price_$><> <price><>   <range2><-> <price><>  <other><>   <other><___entWrdLblOther>   <other><>   <other><___entWrdLblOther>   <other><>  <mileage><>   <units_mileage_mi><>",
    "  <other><___entWrdLblOther>  <other><> <color><>  <other><___entWrdLblOther> <brand><>",
    " <model><> <other><___entWrdLblOther>  <other><> <model><>  <other><___entWrdLblOther> <model><>",
    "<units_price_$><> <price><>  <range2><-> <price><>  <other><___entWrdLblOther>  <other><___entWrdLblOther>  <other><> ",
    "  <other><___entWrdLblOther>  <other><___entWrdLblOther>  <other><___entWrdLblOther>  <other><___entWrdLblOther> ",

    # NN must not always label 1970 - 2024 as belonging to year
    "<color><> <other><___rdmYear>  <other><>  <price><___rdmYear> <range2><> <price><___rdmYear> <units_price_$><>   <less><>  <other><than>  <mileage><___rdmYear> <units_mileage_mi><> ",
    " <mileage><___rdmYear> <units_mileage_mi><>   <price><___rdmYear> <units_price_$><> <other><___rdmYear> ",
    " <other><___rdmYear>  <other><> <units_price_$><> <price><___rdmYear> <more><> <mileage><___rdmYear> <units_mileage_mi><>",
    "  <units_price_$><> <price><___rdmYear> <mileage><___rdmYear> <units_mileage_mi><> <more><> ",
)
specific_segments = (
    # e.g. {'genesis': ['brand', 'model']} the code should do the following:
    #    <other><brand> <brand><genesis>
    #    <other><model> <model><genesis>
    "<TBD><___multilabel> ",  # TBD => To Be Determined
    "<TBD><___assoc_brand_modelNum> ",
)
# "setting_segments" are in "remove segment" because restore in the middle of a
# full sentence auto-removes everything before it
# setting_segments = (
#     "<restore><> <setting><>",
#     "<restore><>",
# )
segments_weights = {
    # weights are relative to each other
    # e.g. brand_probability = brand_weight / (sum of all weights)
    "brand": {
        "segments": brand_segments,
        "weight": 50
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
    "remove_restore": {
        "segments": remove_restore_segments,
        "weight": 5
    },
    "mixed": {
        "segments": mixed_segments,
        "weight": 50
    },
    "specific": {
        "segments": specific_segments,
        "weight": 10
    },
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

# tuple of allowed labels; only these can be used in segments above
include_noNum_labels = (
    "TBD",  # used with ___multilabel and ___assoc_brand_modelNum
    "other",
)
noNum_labels = tuple(
    set(
        tuple(cmdsLbls.keys()) + tuple(unitsLbls.keys()) +
        carEntityNonNumLbls + include_noNum_labels))
include_num_labels = ("setting", )
num_labels = tuple(set(carEntityNumLbls + include_num_labels))
assert set(noNum_labels).intersection(set(num_labels)) == set()
all_labels = noNum_labels + num_labels
del include_noNum_labels
del include_num_labels
del num_labels

# entityWrds_withLbl_other is a tuple of those entityWrds that should be
# labeled based on the context in the sentence; either they have their proper
# label or they have a label of "other"; the entityWrds in this tuple are ONLY
# used in a context where the NN must label them as "other";
# usage: <other><> <other><___entWrdLblOther> <other><>
# Note: ___entWrdLblOther has both numEntityWrds and nonNumEntityWrds; the
# nonNumEntityWrds are in entityWrds_withLbl_other whereas the numEntityWrds
# are those numbers that are used as entityWrds for labels such as year, price,
# mileage, setting; the numEntityWrds are not generated in this file but are
# generated elsewhere by the code
# Note: In contrast, entityWrds of entityLbls such as brand, model, color have
# a fixed label; e.g. entityWrd "red" has the same entityLbl of "color"
# regardless of the context of the sentence
entityWrds_withLbl_other = set()
for entityWrds in synonyms_for_carEntityNonNumLbls.values():
    entityWrds_withLbl_other.update(set(entityWrds))
for entityWrds in cmdsLbls.values():
    entityWrds_withLbl_other.update(set(entityWrds))
entityWrds_withLbl_other.update(set(units))
entityWrds_withLbl_other.update(set(carEntityLbls), set(hyphen))
entityWrds_withLbl_other = list(entityWrds_withLbl_other -
                                set(cmdsLbls["restore"]) -
                                set(cmdsLbls["remove"]))
"""
units               ('$', 'dollar', 'dollars', 'mi', 'mile', 'miles', 'mileage', 'mileages')

carEntityNonNumLbls ('brand', 'model', 'color')
carEntityNumLbls    ('mileage', 'price', 'year')
carEntityLbls       ('brand', 'model', 'color', 'mileage', 'price', 'year')

noNum_labels        ('brand', 'TBD', 'less', 'range2', 'restore', 'units_mileage_mi', 'remove', 'units_price_$', 'color', 'more',
                        'model', 'range1', 'other')
all_labels          ('units_price_$', 'more', 'brand', 'model', 'color', 'TBD', 'range1', 'other', 'units_mileage_mi', 'range2',
                        'remove', 'restore', 'less', 'price', 'year', 'setting', 'mileage')
entityWrds_withLbl_other   ['mile', '-', 'mi', 'range', 'mileages', 'greater', 'brand', 'mileage', 'above', 'over', '$', 'less',
                                'larger', 'miles', 'dollar', 'higher', 'lower', 'price', 'between', 'through', 'under', 'little',
                                'year', 'below', 'make', 'smaller', 'to', 'dollars', 'color', 'more', 'model']
"""
