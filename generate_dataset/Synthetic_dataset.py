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

train_sentences = (
    # user should normally type as follows
    " <color><> <year><> <brand><> <model><>  <mileage><>   <range2><-> <mileage><>  <units_mileage_mi><>   <range1><>   <units_price_$><> <price><> and <price><>  ",
    " <model><>  <year><> <range2><> <year><>   <color><>    <less><> than  <price><> <units_price_$><> and  <less><> than  <mileage><>   <units_mileage_mi><> ",
    " <color><>  <more><>  than <year><>  <model><> <brand><>   <units_price_$><> <price><>   <range2><-> <price><>  <mileage><>   <range2><->  <units_mileage_mi><mi> <mileage><>  ",
    "<brand><> <less><>  <year><>   <color><> <model><>   <range1><>  <mileage><>  <units_mileage_mi><> and <mileage><>    <range1><>   <price><> <price><>  <units_price_$><> ",

    # NN must sometimes NOT label entityWrds; usually (but not always) when they are between two "O" words
    "give <color><> through <brand><>  ",  # In Predict, NN must not label "through" as range2 label
    " run it <year><> <range2><> <year><> and to the ground  <mileage><>   <range2><-> <mileage><>  <units_mileage_mi><> either higher or lower it don't matter  <units_price_$><> <price><>  <range2><-> <price><> because little is known",
    "between the world of  <year><> but larger than  <color><> and smaller than <brand><> lies <price><>  <range2><-> <price><> <units_price_$><> ",
    "it's more  <brand><> and above <model><> but through below the price that is not right <less><> than  <units_price_$><> <price><>",
    " under the <model><>  but over the  <brand><> in the range of  <color><> for <less><> than  <mileage><>   <units_mileage_mi><> ",
    "your mileage and $ or dollar is less but prices are greater so I will give <price><> <units_price_$><>",
    "the range of mileages and dollars is above or below but i say <units_price_$><> <price><>   <range2><-> <price><> which is neither greater or little to whatcha say <mileage><>   <units_mileage_mi><>",

    # NN must not always label 1970 - 2024 as belonging to year
    "<color><> <year><1992> <brand><> <price><1992> <range2><> <price><2020> <units_price_$><>  with <less><less> than <mileage><1992> <units_mileage_mi><miles> ",
    "with <mileage><2016> <units_mileage_mi><miles> and <price><2016> <units_price_$><> <year><2016> or <more><more>",
    " <brand><> <year><1975> <units_price_$><> <price><1975> <more><more> <mileage><1975> <units_mileage_mi><miles>",
    "<year><1982> <model><>  <units_price_$><> <price><1994> <mileage><1999> <units_mileage_mi><miles> <more><more> ",

    # miscellaneous
    "<year><> <model><>",
    "<year><> <brand><>",
    "<year><> <color><> <model><>",
    "<year><> <color><> <brand><>",
)
val_sentences = (
    # user should normally type as follows
    "<mileage><>   <range2><-> <mileage><>  <units_mileage_mi><>   <range1><>   <units_price_$><> <price><> and <price><>  <year><> <color><> <brand><> <model><>  ",
    "  <year><> <range2><> <year><>   <color><>    <less><> than  <units_price_$><> <price><> <model><>   <less><> than  <mileage><>   <units_mileage_mi><> ",
    " <color><>  <more><>  than <year><>  <model><>  <units_price_$><> <price><>   <range2><-> <price><>  <mileage><>   <range2><->  <units_mileage_mi><mi> <mileage><>   <brand><> ",
    "<brand><>  <color><> <model><>   <range1><>  <mileage><>  and <mileage><>  <units_mileage_mi><>   <range1><>   <units_price_$><> <price><> <price><>   <less><>  <year><> ",

    # NN must sometimes NOT label entityWrds; usually (but not always) when they are between two "O" words
    "<color><> to <model><> and in between the $ and still there are <less><> than  <units_price_$><> <price><> ",
    "i have to tell  that larger than dollars are  <mileage><>   <range2><-> <mileage><>  <units_mileage_mi><> but they are not above this  <year><> <range2><> <year><> and the range is incomplete  <units_price_$><> <price><>   <range2><-> <price><> ",
    "smaller than   <year><> but higher than the usual <price><> <units_price_$><> <color><> and below  <brand><> ",
    "  <brand><> but through i want <less><> than <units_price_$><>   <price><> regardless of greater prices",
    " mileages are under <model><>  when little is known of <price><>   <range2><-> <price><>  <units_price_$><> and less the better for  <mileage><>   <units_mileage_mi><> ",
    "<price><> <units_price_$><> is the price i'll pay for <brand><> which is more  ",
    "over the years the  <price><>   <range2><-> <price><> <units_price_$><> is not lower in dollar for <color><> without the mileage",

    # NN must not always label 1970 - 2024 as belonging to year
    "<model><> <brand><>  <year><2008> <price><2008> <range2><> <price><2008> <units_price_$><>  with  <mileage><2008> <units_mileage_mi><miles> <more><more> ",
    "with <mileage><2016> <units_mileage_mi><miles> and <price><2016> <units_price_$><> <year><2016> or <more><more>",
    " <color><> <year><1975> <units_price_$><> <price><1975> <more><more> <mileage><1975> <units_mileage_mi><miles>",
    "<year><2010> <model><> <mileage><2003> <units_mileage_mi><miles>   <less><less> than  <units_price_$><> <price><1987>",

    # miscellaneous
    "<year><> <model><>",
    "<year><> <brand><>",
    "<year><> <color><> <model><>",
    "<year><> <color><> <brand><>",
)
test_sentences = (
    # user should normally type as follows
    "<year><> <color><> <brand><> <model><> <less><> than  <units_price_$><> <price><> and  <less><> than  <mileage><>   <units_mileage_mi><> ",
    " <color><>  <more><>  than <year><>  <model><> <brand><>  <range1><>  <mileage><>  <units_mileage_mi><> and <mileage><>    <range1><>   <units_price_$><> <price><> <price><>  ",
    "<brand><> <less><>  <year><>   <color><> <model><>   <units_price_$><> <price><>   <range2><-> <price><>  <mileage><>   <range2><->  <units_mileage_mi><mi> <mileage><> ",
    " <model><>  <year><> <range2><> <year><>   <color><>      <mileage><>   <range2><-> <mileage><>  <units_mileage_mi><>   <range1><>   <units_price_$><> <price><> and <price><> ",

    # NN must sometimes NOT label entityWrds; usually (but not always) when they are between two "O" words
    "these dollars and mileages are not worth <brand><> through <brand><>  ",  # In Predict, NN must not label "through" as range2 label
    "i have to tell  that $ is not <year><> <range2><> <year><>   <mileage><>   <range2><-> <mileage><>  <units_mileage_mi><>  <units_price_$><> <price><>   <range2><-> <price><>  is between a few cars i want through <model><> ",
    "more or less it is dollars that count  <year><> but larger than  <color><> and smaller than <brand><> ",
    "it's between  <brand><> and <brand><> but through and through i want <less><> than  <units_price_$><> <price><>",
    " <model><>  is under  but  <brand><> is over  <color><> for <less><> than  <mileage><>   <units_mileage_mi><> ",
    "your price is higher but mileage is lower so I will give <price><> <units_price_$><>",
    "the range of prices is above or below but i say <price><>   <range2><-> <price><> <units_price_$><> which is neither greater or little",

    # NN must not always label 1970 - 2024 as belonging to year
    "<year><2020> <model><> with <less><less> than <mileage><2020> <units_mileage_mi><miles> <color><> <price><2015> <range2><> <price><2020> <units_price_$><>",
    "with <mileage><2015> <units_mileage_mi><miles> <year><2015> or <more><more> and <price><2015> <units_price_$><>",
    "<year><2023> <units_price_$><> <price><2023> <more><more> <mileage><2023> <units_mileage_mi><miles>",
    "<year><2010> <units_price_$><> <price><2015> <mileage><2017> <units_mileage_mi><miles> <more><more> ",

    # miscellaneous
    "<units_price_$><$> <price><> or <units_price_$><$> <price><> <range2><-> <price><>",
    "  <range1><>  <price><> and <price><>  price ",
)
full_sentences = {
    "train": train_sentences,
    "val": val_sentences,
    "test": test_sentences
}

brand_segments = (
    # neural-net must memorize, e.g., toyota is brand, without any hints
    'gimme just the top "reliable" cars  ',
    "<brand><>  ",
    "<brand><>  ",
    "<brand><>  ",
    "<brand><>  ",
    "<brand><>  ",
    "<brand><>  ",
    "<brand><>  ",
    "<brand><>  ",
    "<brand><>  ",
    "<brand><>  ",
    "<brand><>  ",
    "<brand><>  ",
    "<brand><>  ",
    "<brand><>  ",
    "<brand><>  ",
    "<brand><>  ",
    "<brand><>  ",
    "<brand><>  ",
    "<brand><>  ",
    "<brand><>  ",
    "<brand><>  ",
    "<brand><>  ",
    "<brand><>  ",
    "<brand><>  ",
    "<brand><>  ",
    "<brand><>  ",
    "<brand><>  ",
    "<brand><>  ",
    "<brand><>  ",
    "<brand><>  ",
    " <brand><> <brand><> <brand><>  ",
    " I want to buy <brand><> ",
    " the best brand is <brand><> so that is what I want",
    " give me <brand><>  for life",
    "brand <brand><genesis>  ",
    " <brand><genesis> brand ",
)
model_segments = (
    # neural-net must memorize, e.g., camry is model, without any hints
    "i like expensive models  ",
    "<model><>  ",
    "<model><>  ",
    "<model><>  ",
    "<model><>  ",
    "<model><>  ",
    "<model><>  ",
    "<model><>  ",
    "<model><>  ",
    "<model><>  ",
    "<model><>  ",
    "<model><>  ",
    "<model><>  ",
    "<model><>  ",
    "<model><>  ",
    "<model><>  ",
    "<model><>  ",
    "<model><>  ",
    "<model><>  ",
    "<model><>  ",
    "<model><>  ",
    "<model><>  ",
    "<model><>  ",
    "<model><>  ",
    "<model><>  ",
    "<model><>  ",
    "<model><>  ",
    "<model><>  ",
    "<model><>  ",
    "<model><>  ",
    "<model><>  ",
    "<model><>  ",
    "<model><>  ",
    "<model><>  ",
    "<model><>  ",
    "<model><>  ",
    "<model><>  ",
    "<model><>  ",
    "<model><>  ",
    "<model><>  ",
    "<model><>  ",
    "<model><>  ",
    "<model><>  ",
    "<model><>  ",
    "<model><>  ",
    "<model><>  ",
    "<model><>  ",
    "<model><>  ",
    "<model><>  ",
    "<model><>  ",
    "<model><>  ",
    "<model><>  ",
    "<model><>  ",
    "<model><>  ",
    "<model><>  ",
    "<model><>  ",
    "<model><>  ",
    "<model><>  ",
    "<model><>  ",
    "<model><>  ",
    "<model><>  ",
    "<model><>  ",
    "<model><>  ",
    "<model><>  ",
    "<model><>  ",
    "<model><>  ",
    "<model><>  ",
    "<model><>  ",
    "<model><>  ",
    "<model><>  ",
    "<model><>  ",
    "<model><>  ",
    "<model><>  ",
    "<model><>  ",
    "<model><>  ",
    "<model><>  ",
    "<model><>  ",
    "<model><>  ",
    "<model><>  ",
    " I love <model><> ",
    " not the worst model is <model><> and that is what I want",
    " show me <model><>  for all i care",
    "  <model><>  <model><>  <model><>   ",
    "model <model><genesis>  ",
    " <model><genesis> model ",
)
color_segments = (
    # neural-net must memorize, e.g., red is color, without any hints
    "most popular only, please  ",
    "most popular color is <color><> and that is for me ",
    "<color><> is best ",
    "<color><> is the way to go ",
    "<color><>  ",
    "<color><> <color><>  ",
    "<color><> or <color><> or <color><>  ",
)
price_segments = (
    # a number, e.g. 2022 by itself does not tell whether user refers to price,
    # year, or mileage; so a hint is needed, for ,e.g., a $ sign

    "what is 'the' price of \"rice\" in china  ",

    # " <price><>",
    "<units_price_$><$> <price><>",
    "<price><> <units_price_$><dollars>",
    "<units_price_$><> <price><> for car",
    "i will pay <price><> <units_price_$><>",
    "me thinks <units_price_$><> <price><> is the way to go ",
    "how about <price><> <units_price_$><> for this junk",
    "<units_price_$><$> <price><> <price><> <units_price_$><dollars> ",

    # "  <price><> <range2><-> <price><> ",
    " <units_price_$><> <price><>   <range2><-> <price><> ",
    " <price><>   <units_price_$><$>  <range2><-> <price><> ",
    " <price><>   <range2><->  <units_price_$><$> <price><> ",
    " <price><>   <range2><-> <price><>  <units_price_$><> ",
    " <units_price_$><> <price><>   <range2><-> <price><> as an example",
    " my car should be  <price><>   <range2><-> <price><> <units_price_$><>",
    " somewhere in  <price><>   <range2><-> <price><> <units_price_$><> is where the deal lies",

    # " <range1><>  <price><> <price><>  ",
    " <range1><>   <units_price_$><> <price><> <price><>  ",
    " i'll say <range1><>   <units_price_$><> <price><> and <price><> is the way to go ",
    " <range1><>  <price><>  <units_price_$><> <price><>  ",
    " <range1><>  <price><> <price><>   <units_price_$><> ",
    " <range1><> of  <units_price_$><> <price><> and <price><> is the right price ",

    # " <less><>  <price><>  ",
    " <less><>   <units_price_$><> <price><>  ",
    " <less><>  <price><>   <units_price_$><> ",
    " it should be <less><>   <price><>   <units_price_$><> for the sale to happen",
    " <less><> than  <units_price_$><> <price><> is what i will pay  ",
    # "  <price><>  <less><> ",
    "  <units_price_$><>  <price><>  <less><> ",
    " the way to go is  <units_price_$><>  <price><>  <less><> as far as i am concerned",
    "  <price><>   <units_price_$><> <less><> ",
    "  the only way is <price><>   <units_price_$><> or  <less><> ",

    # " <more><>  <price><>  ",
    " <more><>   <units_price_$><> <price><>  ",
    " <more><>  <price><>   <units_price_$><> ",
    "me thinks  <more><>  <price><>   <units_price_$><> will kill the deal",
    " <more><> than  <units_price_$><> <price><> is the way to go  ",
    # "  <price><>  <more><> ",
    "  <units_price_$><>  <price><>  <more><> ",
    "you think  <units_price_$><>  <price><>  <more><> is what it will take",
    "  <price><>   <units_price_$><> <more><> ",
    "  the  way it is done is <price><>   <units_price_$><> or  <more><> ",
)
mileage_segments = (
    # a number, e.g. 2022 by itself does not tell whether user refers to price,
    # year, or mileage; so a hint is needed, for ,e.g., the word "miles" in the
    # sentence

    "i want 'low' in \"miles\"  ",

    # " <mileage><>",
    "<units_mileage_mi><> <mileage><>",
    "<mileage><> <units_mileage_mi><> ",
    "it should be <mileage><> <units_mileage_mi><> ",
    "<units_mileage_mi><> <mileage><> is the wish",
    "<units_mileage_mi><> <mileage><> <mileage><> <units_mileage_mi><> ",

    # "  <mileage><> <range2><-> <mileage><> ",
    " <units_mileage_mi><> <mileage><>   <range2><-> <mileage><> ",
    " <mileage><>   <units_mileage_mi><mi>  <range2><-> <mileage><> ",
    " <mileage><>   <range2><->  <units_mileage_mi><mi> <mileage><> ",
    " <mileage><>   <range2><-> <mileage><>  <units_mileage_mi><> ",
    " <units_mileage_mi><> <mileage><>   <range2><-> <mileage><> will work for me",
    " my car should be  <mileage><>   <range2><-> <mileage><> <units_mileage_mi><>",

    # " <range1><>  <mileage><> <mileage><>  ",
    " <range1><>   <units_mileage_mi><> <mileage><> <mileage><>  ",
    " <range1><>  <mileage><>  <units_mileage_mi><> <mileage><>  ",
    " <range1><>  <mileage><> <mileage><>   <units_mileage_mi><> ",
    " <range1><> of  <units_mileage_mi><> <mileage><> and <mileage><> is the right price ",

    # " <less><>  <mileage><>  ",
    " <less><>   <units_mileage_mi><> <mileage><>  ",
    " <less><>  <mileage><>   <units_mileage_mi><> ",
    " <less><> than  <units_mileage_mi><> <mileage><> is what i will pay  ",
    # "  <mileage><>  <less><> ",
    "  <units_mileage_mi><>  <mileage><>  <less><> ",
    "  <mileage><>   <units_mileage_mi><> <less><> ",
    "  the only way is <mileage><>   <units_mileage_mi><> or  <less><> ",

    # " <more><>  <mileage><>  ",
    " <more><>   <units_mileage_mi><> <mileage><>  ",
    " <more><>  <mileage><>   <units_mileage_mi><> ",
    " <more><> than  <units_mileage_mi><> <mileage><> is the way to go  ",
    # "  <mileage><>  <more><> ",
    "  <units_mileage_mi><>  <mileage><>  <more><> ",
    "  <mileage><>   <units_mileage_mi><> <more><> ",
    "  the  way it is done is <mileage><>   <units_mileage_mi><> or  <more><> ",
)
year_segments = (
    # a number, e.g. 2022 by itself does not tell whether user refers to price,
    # year, or mileage; so a hint is needed, for ,e.g., the word "year" in the
    # sentence. ***Actually I want a number between 1950 - 2023 without units
    # to be recognized by the model with a label of year
    # IMPORTANT: The following sentence does not have a hint of "year" but
    # the context contains brand_value/model_value which the model must learn:
    # "i want <year><2023> red toyota  ",

    "just the latest years  ",

    # " <year><>",
    " <year><>",
    " year <year><> ",
    " <year><> year",
    " year <year><> is the way",
    " the best were <year><> <year><> and that is what i want",

    # " <year><> <range2><-> <year><>  ",
    " <year><> <range2><> <year><>  ",
    " year <year><> <range2><> <year><>  ",
    " <year><> <range2><> <year><> years ",
    " for this reason <year><> <range2><> <year><> are the best ",

    # " <range1><>  <year><> <year><>  ",
    " <range1><>  <year><> <year><>  ",
    " <range1><>   year <year><> <year><>  ",
    " <range1><>  <year><>  year <year><>  ",
    " <range1><>  <year><> <year><>   year ",
    " <range1><>  <year><> <year><> are the best ",

    # " <less><>  <year><>  ",
    " <less><>  <year><>  ",
    " <less><>  year <year><>  ",
    " <less><>  <year><>  year ",
    " <less><> than   <year><> is what matters  ",
    # "  <year><>  <less><> ",
    "  <year><>  <less><> ",
    "  year  <year><>  <less><> ",
    "  <year><>   year <less><> ",
    "  the only way is <year><>   or  <less><> ",

    # " <more><>  <year><>  ",
    " <more><>  <year><>  ",
    " <more><>   year <year><>  ",
    " <more><>  <year><>   year ",
    " <more><> than   <year><> is the way to go  ",
    # "  <year><>  <more><> ",
    "  <year><>  <more><> ",
    "  year <year><>  <more><> ",
    "  <year><>   year <more><> ",
    "  ok, i like <year><>  or  <more><> ",
)
remove_segments = (
    "<remove><> <everything><>",
    "<remove><> <restore><> <remove><> ",
    "<remove><> <brand><>  <brand><toyota> <model><> <model><cruiser>  <color><>  <color><white> ",
    "<remove><> <units_price_$><> <price><> <units_mileage_mi><> <mileage><> <year><> ",
    "<restore><> <remove><> <model><nv3500> <mileage><26632.01> <units_price_$><$> <price><5000>",
    "<restore><>  <setting><> <remove><> <brand><bentley> <mileage><2987> <units_mileage_mi><miles> <price><5000> <year><2018> ",
    "<restore><>  <setting><> <remove><>  <color><metallic>  <year><2018> ",
    "<restore><>  <setting><145> <remove><> <model><passenger> <color><tuxedo> <mileage><2987> <price><5000> <units_price_$><> <year><2018> ",
    "<restore><>  <setting><15> <remove><> <model><hd> <color><white>  <units_mileage_mi><mile> <mileage><2987> <units_price_$><>  <price><5000> <year><2024> ",
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
# full sentence makes everything before it redundant
# setting_segments = (
#     "<restore><> <setting><>",
#     "<restore><>",
# )
entityLbls_mapTo_segments = {
    "brand": brand_segments,
    "model": model_segments,
    "color": color_segments,
    "price": price_segments,
    "year": year_segments,
    "mileage": mileage_segments,
    "remove": remove_segments,
    # "style": style_segments,
    # "setting": setting_segments,
}

cmdsLbls = {
    "restore": ("restore", "retrieve", "retreive",),
    "more": ("over", "more", "higher", "above", "larger", "greater"),
    "less": ("under", "less", "lower", "below", "smaller", "little"),
    "range1": ("between", "range"),  # e.g. between $5000 and $7000
    "range2": ("-", "to", "through"),  # e.g. $5000 - $7000
    "remove": ("remove", "delete"),
    "everything": ("everything", ),
}
# carEntityNum must precede following commands
cmdsLbls_follow_carEntityNum = ("range2", )

unitsLbls = {
    "units_price_$": ("$", "dollar", "dollars", "price", "prices",),
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
    "style",    # needed in Fill_entityWrds.py and deleted
)
synonyms_for_carEntityNonNumLbls = {
    # different datasets use different names, e.g. "make" instead of "brand"
    "brand": ["brand", "make"],
    "model": ["model"],
    "color": ["color"],
    "style": ["body_style", "body_styles"],
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
        return str(round(random.uniform(0, 300000), 2))
    else:
        return str(int(random.uniform(0, 300000)))


def price_func() -> str:
    if random.getrandbits(1):
        return str(round(random.uniform(0, 3000000), 2))
    else:
        return str(int(random.uniform(0, 3000000)))


def setting_func() -> str:
    return str(random.randint(1, 30))


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
