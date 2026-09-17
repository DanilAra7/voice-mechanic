from selectolax.parser import HTMLParser

from mechanic.data.common import looks_english
from mechanic.data.stackexchange import match_vehicles, parse_tags
from mechanic.data.startmycar import parse_card, parse_comments

CARD = """
<div class="solucionado js-report" id="den53376" data-denunciaid="53376"><div class="Report">
  <h3 class="TitleListing"><a href="/us/honda/accord/problems/53376/intermittent-stalling">Intermittent stalling</a>
    <span>Solved</span></h3>
  <div class="text-ellipsis text-normal color-text-gray">Honda Accord 2014 &nbsp;120000 miles</div>
  <div class="TagList"><a class="TagLink">Stalls</a><a class="TagLink">Brakes</a></div>
  <div class="js-report-body Text">Car stalls randomly while driving. No codes found by mechanic.</div>
  <span>1 reply&nbsp;</span>
</div></div>
"""

COMMENTS = """
<div class="CommentCard--solution CommentCard">
  <div class="CommentCard__text Text"><span>Replaced the IACV.</span></div></div>
<div class="CommentCard"><div class="CommentCard__text Text"><span>Check the crank sensor.</span></div></div>
"""


def test_looks_english():
    assert looks_english("The car stalls when I turn the wheel and the engine light is on")
    assert not looks_english("Se apaga al acelerar y el motor hace ruido")
    assert not looks_english("هل تيربو ضروري")
    assert not looks_english("ok")


def test_parse_startmycar_card():
    item = parse_card(HTMLParser(CARD).css_first("div.js-report"))
    assert item["id"] == "53376"
    assert item["title"] == "Intermittent stalling"
    assert item["year"] == 2014
    assert item["tags"] == ["Stalls", "Brakes"]
    assert item["solved"] and item["replies"] == 1
    assert item["url"].endswith("/53376/intermittent-stalling")


def test_parse_startmycar_comments():
    assert parse_comments(COMMENTS) == [(True, "Replaced the IACV."), (False, "Check the crank sensor.")]


def test_stackexchange_tags_and_vehicle_match():
    assert parse_tags("|honda|civic|check-engine|") == ["honda", "civic", "check-engine"]
    assert parse_tags("<audi><a4>") == ["audi", "a4"]
    assert match_vehicles(["honda", "civic"], None) == ["honda_civic_10"]
    assert match_vehicles(["honda", "civic"], 2005) == []
    assert match_vehicles(["audi", "a4"], 2012) == ["audi_a4_b8"]
