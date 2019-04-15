import unittest
import Classifier as cs
import Magic as mgc


class TestSum(unittest.TestCase):

    def test_getSeriesDataCorrectly(self):
        X, gene_ids, labels, titles = cs.get_series_data("unitFiles/smallTestFile.txt", 3, 0, 'true', 'true')
        self.assertEqual(labels, ['label','treg','treg','treg','treg','treg','treg','treg','treg','treg','treg','treg','treg','treg','treg','treg','treg','treg','treg'], "Should parse the file labels correctly")
        self.assertEqual(titles, ['amk3_tumor_treg#1', 'amk5_tumor_treg#2', 'amk6_tumor_treg#3', 'amk5_normal_treg#1', 'amk6_normal_treg#2', 'amk7_normal_treg#3', 'amk7_tumor_treg#4', 'amk9_tumor_treg#6', 'amk10_tumor_treg#7', 'amk11_tumor_treg#8', 'amk12_tumor_treg#9', 'amk13_tumor_treg#10', 'amk14_tumor_treg#11', 'amk15_tumor_treg#12', 'amk8_normal_treg#4', 'amk10_normal_treg#5', 'amk11_normal_treg#6', 'amk15_normal_treg#9'],
              "Should parse the file titles correctly")

        self.assertEqual(X.shape, (21, 18), "shape should be 21 genes and 18 cells")

    def test_getSeriesDataERROR(self):
        self.assertEqual(cs.get_series_data("unitFiles/empty.txt", 3, 0, 'true', 'true'),
                         'Empty',
                         "Should return Empty on an empty file")

    def test_gene_code_map_Correctly(self):
        id_to_sym=cs.gene_code_map("unitFiles/smallMapFile.txt", 2, 0, 1, 2)
        self.assertEqual(id_to_sym,
                         {'ddr1': '1007_s_at', 'rfc2': '1053_at', 'hspa6': '117_at', 'pax8': '121_at', 'guca1a': '1255_g_at', 'mir5193': '1294_at', 'thra': '1316_at'},
                         "Should return 7 genes symbols")

    def test_gene_code_map_ERROR(self):
        with self.assertRaises(SyntaxError): #should raise error on files with no colums- not the right format
            cs.gene_code_map("unitFiles/nocol.txt", 3, 0, 1, 2)

    def test_gene_code_map_empty(self):
        self.assertEqual(cs.gene_code_map("unitFiles/empty.txt", 3, 0, 1, 2), {},
                    "Should return empty dictionary on an empty file")

    def test_magic_process_fail_on_file_with_no_cols(self):
        with self.assertRaises(SyntaxError): #should raise error on files with no colums- not the right format
            mgc.magic_process("unitFiles/nocol.txt", 3)

    def test_magic_process_fail_on_microarray_cell_type(self):
        with self.assertRaises(EOFError):
            x,y= mgc.magic_process("unitFiles/smallTestFile.txt", 3)

    def test_magic_process_success_on_single_cell_type(self):
        test_data, gene_ids = mgc.magic_process("unitFiles/GSE60781_single_cell_dataset.txt", 3)
        self.assertIsNotNone(test_data, "should return results after magic process")
        self.assertIsNotNone(gene_ids, "should return results after magic process")



if __name__ == '__main__':
    unittest.main()