from eosin.parser import Parser
import pytest
from os import system
from pandas import DataFrame


@pytest.fixture
def parser():
    parser = Parser("./tests/test_pdfs/test.pdf")
    system("pwd")
    return parser


def test_parser_pdf(parser):
    assert type(parser.parse()), DataFrame
