import unittest
from aws_connect.main import main


class ThisTest(unittest.TestCase):
    def test_code(self):
        main()


if __name__ == '__main__':
    unittest.main()
