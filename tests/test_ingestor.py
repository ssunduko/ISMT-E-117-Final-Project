import unittest

from utils import ingestor

class Tests(unittest.TestCase):

    def test_read_first_two_lines_succeeds(self):
        doc = ingestor.read_file(ingestor.TRAINING_FILE)
        self.assertEqual(9119, len(doc))
        self.assertEqual("1", doc[0][0])
        self.assertEqual(doc[0][1],"my life is meaningless i just want to end my life so badly my life is completely empty and i dont want to have to create meaning in it creating meaning is pain how long will i hold back the urge to run my car head first into the next person coming the opposite way when will i stop feeling jealous of tragic characters like gomer pile for the swift end they were able to bring to their lives")
        self.assertEqual("1", doc[1][0])
        self.assertEqual(doc[1][1],"muttering i wanna die to myself daily for a few months now i feel worthless shes my soulmate i cant live in this horrible world without her i am so lonely i wish i could just turn off the part of my brain that feels ")


if __name__ == "__main__":
    unittest.main()