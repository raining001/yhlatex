from Levenshtein import *
from warnings import warn

class StringMatcher:
    """A SequenceMatcher-like class built on the top of Levenshtein"""

    def _reset_cache(self):
        self._ratio = self._distance = None
        self._opcodes = self._editops = self._matching_blocks = None

    def __init__(self, isjunk=None, seq1='', seq2=''):

        if isjunk:
            warn("isjunk not NOT implemented, it will be ignored")
        self._str1, self._str2 = seq1, seq2
        self._reset_cache()

    def set_seqs(self, seq1, seq2):
        self._str1, self._str2 = seq1, seq2
        self._reset_cache()

    def set_seq1(self, seq1):
        self._str1 = seq1
        self._reset_cache()

    def set_seq2(self, seq2):
        self._str2 = seq2
        self._reset_cache()

    # 给出所有第一个字符串转换成第二个字符串需要权重的操作和操作详情会给出一个列表，列表的值为元祖，每个元祖中有5个值
    # [('delete', 0, 1, 0, 0), ('equal', 1, 3, 0, 2), ('insert', 3, 3, 2, 3), ('replace', 3, 4, 3, 4)]
    # 第一个值是需要修改的权重，例如第一个元祖是要删除的操作,2和3是第一个字符串需要改变的切片起始位和结束位，例如第一个元祖是删除第一字符串的0-1这个下标的元素
    # 4和5是第二个字符串需要改变的切片起始位和结束位，例如第一个元祖是删除第一字符串的0-0这个下标的元素，所以第二个不需要删除
    def get_opcodes(self):
        if not self._opcodes:
            if self._editops:
                self._opcodes = opcodes(self._editops, self._str1, self._str2)
            else:
                self._opcodes = opcodes(self._str1, self._str2)
        return self._opcodes



    def get_editops(self):
        if not self._editops:
            if self._opcodes:
                self._editops = editops(self._opcodes, self._str1, self._str2)
            else:
                self._editops = editops(self._str1, self._str2)
        return self._editops

    def get_matching_blocks(self):
        if not self._matching_blocks:
            self._matching_blocks = matching_blocks(self.get_opcodes(),
                                                    self._str1, self._str2)
        return self._matching_blocks

    def ratio(self):
        if not self._ratio:
            self._ratio = ratio(self._str1, self._str2)
        return self._ratio

    def quick_ratio(self):
        # This is usually quick enough :o)
        if not self._ratio:
            self._ratio = ratio(self._str1, self._str2)
        return self._ratio

    def real_quick_ratio(self):
        len1, len2 = len(self._str1), len(self._str2)
        return 2.0 * min(len1, len2) / (len1 + len2)

    def distance(self):
        if not self._distance:
            self._distance = distance(self._str1, self._str2)
        return self._distance
