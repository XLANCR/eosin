import pytest
from os import system
from pandas import DataFrame

try:
    from eosin.parser import Parser
except ModuleNotFoundError:
    pytest.skip("legacy eosin.parser was retired by the evidence-service migration", allow_module_level=True)


@pytest.fixture
def parser():
    parser = Parser("./tests/test_pdfs/test.pdf")
    system("pwd")
    return parser


def test_parser_pdf(parser):
    assert type(parser.parse()), DataFrame
