
class SplitTypes(object):
    Train = 0
    Valid = 1
    Test = 2

    split_types_collection = [Train, Valid, Test]

    @staticmethod
    def get_split_type_name(split_type):
        if split_type == SplitTypes.Train:
            return "Train"
        if split_type == SplitTypes.Valid:
            return "Valid"
        if split_type == SplitTypes.Test:
            return "Test"
