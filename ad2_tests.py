import unittest
from autodrummer2 import sort
from autodrummer2 import find_peaks



class TestCase(unittest.TestCase):

    def test_find_peaks(self):
        pass


    def test_sort(self):
        list1 = [1, 7, 4, 6]
        self.assertEqual(sort(list1, None, None, False), [1, 4, 6, 7])
        list1 = ["a", "f", "d"]
        self.assertEqual(sort(list1, None, None, True), ["f", "d", "a"])
        list1 = [[5, 5], [0, 6], [7, 2]]
        self.assertEqual(sort(list1, 1, None, False), [[7, 2], [5, 5], [0, 6]])


        


def main():
    # execute unit tests
    unittest.main()

if __name__ == '__main__':
    # execute main() function
    main()
