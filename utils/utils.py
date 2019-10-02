import itertools
import json
import math
from json import JSONEncoder

from future.utils import with_metaclass


# https://stackoverflow.com/questions/10252010/serializing-class-instance-to-json
class JsonSerialization:
    """
    own json serializer for mbit, bit and milliseconds class
    json serializable classes need to be derived from RegisteredJsonSerializer and handle the registration automatically
    """
    registered_classes = []
    registered_classes_dict = {}

    @staticmethod
    def addClass(classname, cls):
        """
        register class for serialization
        :param classname: classname used in json
        :param cls: registered class
        """
        JsonSerialization.registered_classes.append((classname, cls))
        JsonSerialization.registered_classes_dict[classname] = cls

    @staticmethod
    def getClass(classname):
        """
        get class from classname
        :param classname:
        :return: class or None
        """
        if classname not in JsonSerialization.registered_classes_dict:
            return None
        return JsonSerialization.registered_classes_dict[classname]

    @staticmethod
    def dumps(d):
        """
        dumps python objects to string
        :param d: python objects
        :return: json string
        """

        class MyEncoder(JSONEncoder):
            def default(self, o):
                for name, t in JsonSerialization.registered_classes:
                    # is object o part of serializable classes? then add own serialization data
                    if type(o) is t:
                        return {'__class__': name, '__data__': o.toJson()}
                    # otherwise let JSONEncoder handle it
                return super().default(o)

        return json.dumps(d, cls=MyEncoder, sort_keys=True)

    @staticmethod
    def loads(f):
        def decode_classes(dct):
            if '__class__' in dct:
                # looks like it could be a class which was serialized by us
                classname, data = dct['__class__'], dct['__data__']
                # lookup class
                cls = JsonSerialization.getClass(classname)
                if cls is not None:
                    # instantiate from data
                    return cls.fromJson(data)
            return dct

        return json.loads(f, object_hook=decode_classes)


# https://stackoverflow.com/questions/10252010/serializing-class-instance-to-json
class RegisterJson(type):
    """
    MetaClass for RegisteredJsonSerializer
    """

    def __new__(cls, clsname, superclasses, attributedict):
        """
        called for every new class - we therfore install a hook to register the class here
        """
        c = type.__new__(cls, clsname, superclasses, attributedict)
        JsonSerialization.addClass(c.__name__, c)
        return c


# https://stackoverflow.com/questions/10252010/serializing-class-instance-to-json
# https://python-future.org/compatible_idioms.html
class RegisteredJsonSerializer(with_metaclass(RegisterJson)):
    """
    base class for json serializable classes
    """

    def __init__(self, *args):
        self.args = args

    def toJson(self):
        """
        represent class attributes
        :return: class attributes
        """
        return self.args

    @classmethod
    def fromJson(cls, args):
        """
        reinstantiate class with attributes
        :param args: class attributes
        :return: class instance
        """
        return cls(*args)

    def __repr__(self):
        return 'JSONSerializer ' + str(self.args)


class second(RegisteredJsonSerializer):
    """
    SI class second
    """

    def __hash__(self):
        return hash(self.time)

    def __init__(self, time):
        RegisteredJsonSerializer.__init__(self, time)
        self.time = time

    def toMiliseconds(self):
        return self.time * 1000

    def toSeconds(self):
        return self.time

    def __eq__(self, other):
        return other.toSeconds() == self.toSeconds()

    def __ge__(self, other):
        return self.__eq__(other) or self.__gt__(other)

    def __le__(self, other):
        return self.__eq__(other) or self.__lt__(other)

    def __gt__(self, other):
        return self.toSeconds() > other.toSeconds()

    def __lt__(self, other):
        return self.toSeconds() < other.toSeconds()

    def __str__(self):
        return '%ds' % (self.time)

    def __repr__(self):
        return str(self)


class milliseconds(second):
    """
    class for milliseconds
    """

    def __init__(self, time):
        second.__init__(self, time / 1000.0)
        RegisteredJsonSerializer.__init__(self, time)

    def __str__(self):
        return '%dms' % (self.time * 1000.0)


def get_correct_class(inp):
    """
    SI text to class
    :param inp: input text (e.g. "10ms")
    :return: class instance (e.g. milliseconds(10))
    """
    if "ms" in inp:
        return milliseconds(int(inp[:-2]))
    if "s" in inp:
        return second(int(inp[:-1]))
    if "mbit" in inp:
        return mbit(int(inp[:-4]))
    if "bit" in inp:
        return bit(int(inp[:-3]))
    if "," in inp:
        return inp.split(",")
    return inp


class bit(RegisteredJsonSerializer):
    """
    class for a bit
    """

    def __hash__(self):
        return hash(self.size)

    def __eq__(self, other):
        return other.toBit() == self.toBit()

    def __gt__(self, other):
        return self.toBit() > other.toBit()

    def __lt__(self, other):
        return self.toBit() < other.toBit()

    def __ge__(self, other):
        return self.__eq__(other) or self.__gt__(other)

    def __le__(self, other):
        return self.__eq__(other) or self.__lt__(other)

    def __init__(self, size):
        RegisteredJsonSerializer.__init__(self, size)
        self.size = size

    def toBit(self):
        return self.size

    def toByte(self):
        return self.size / 8.0

    def toFullByte(self):
        return math.ceil(self.toByte())

    def toMbit(self):
        return self.size / (1000 * 1000.0)

    def toFullMbit(self):
        return math.ceil(self.size / (1000 * 1000.0))

    def __repr__(self):
        return str(self)

    def __str__(self):
        return '%dbit' % (self.size)


class mbit(bit):
    """
    class for a megabit
    """

    def __init__(self, size):
        bit.__init__(self, size * 1000 * 1000)
        RegisteredJsonSerializer.__init__(self, size)

    def __truediv__(self, other):
        return mbit(self.size / 1000. / 1000. / other)

    def __div__(self, other):
        return self.__truediv__(other)

    def __str__(self):
        return ('%f' % (self.size / 1000. / 1000.)).rstrip("0").rstrip(".") + 'mbit'


def params_gen(dct):
    """
    generate product of network topology parameters
    :param dct:
    :return:
    """
    k = dct.keys()
    v = dct.values()
    return map(lambda x: dict(zip(k, x)), itertools.product(*v))


def dict_to_filename(d):
    """
    flatten dictionary into a string
    :param d: dictionary
    :return: string of dictionary entries
    """
    items = d.items()
    items = sorted(items, key=lambda x: x[0])
    out = []
    for k, v in items:
        out.append("%s_%s" % (str(k), str(v)))
    return ",".join(out)
