  
"""
Provide a base class for cluster objects in CogAlg.
Features:
- Unique instance ids per class.
- Instances are retrievable by ids via class.
- Reduced memory usage compared to using dict.
- Methods generated via string templates so overheads caused by
differences in interfaces are mostly eliminated.
- Can be extended/modified further to support main implementation better.
"""

import weakref
from numbers import Number
from inspect import isclass

NoneType = type(None)

# ----------------------------------------------------------------------------
# Template for class method generation
_methods_template = '''
@property
def id(self):
    return self._id
    
def pack(self{pack_args}):
    """Pack all fields/params back into {typename}."""
    {pack_assignments}
    
def unpack(self):
    """Unpack all fields/params back into the cluster."""
    return ({param_vals})
def accumulate(self, **kwargs):
    """Add a number to specified numerical fields/params."""
    {accumulations}
def __contains__(self, item):
    return (item in {params})
def __delattr__(self, item):
    raise AttributeError("cannot delete attribute from "
                         "'{typename}' object")
def __repr__(self):
    return "{typename}({repr_fmt})" % ({numeric_param_vals})
'''

# ----------------------------------------------------------------------------
# MetaCluster meta-class
class MetaCluster(type):
    """
    Serve as a factory for creating new cluster classes.
    """
    def __new__(mcs, typename, bases, attrs):  # called right before a new class is created
        # get fields/params and numeric params
        replace = attrs.get('replace', {})

        # inherit params
        for base in bases:
            if issubclass(base, ClusterStructure):
                for param in base.numeric_params:
                    if param not in attrs:  # prevents duplication of base params
                        if param in replace:
                            new_param, new_type = replace[param]
                            if new_param is not None:
                                attrs[new_param] = new_type
                        else:
                            attrs[param] = Cdm

        # only ignore param names start with double underscore
        params = tuple(attr for attr in attrs
                       if not attr.startswith('__') and
                       isclass(attrs[attr]))

        numeric_params = tuple(param for param in params
                               if (issubclass(attrs[param], Number)) and
                               not (issubclass(attrs[param], bool))) # avoid accumulate bool, which is flag

        # Fill in the template
        methods_definitions = _methods_template.format(
            typename=typename,
            params=str(params),
            param_vals=', '.join(f'self.{param}'
                                 for param in params),
            numeric_param_vals=', '.join(f'self.{param}'
                                         for param in numeric_params),
            pack_args=', '.join(param for param in ('', *params)),
            pack_assignments='; '.join(f'self.{param} = {param}'
                                  for param in params)
                             if params else 'pass',
            accumulations='; '.join(f"self.{param} += "
                                    f"kwargs.get('{param}', 0)"
                                    for param in numeric_params)
                          if params else 'pass',
            repr_fmt=', '.join(f'{param}=%r' for param in numeric_params),
        )
        # Generate methods
        namespace = dict(print=print)
        exec(methods_definitions, namespace)
        # Replace irrelevant names
        namespace.pop('__builtins__')
        namespace.pop('print')

        # Update to attrs
        attrs.update(namespace)

        # Save default types for fields/params
        for param in params:
            attrs[param + '_type'] = attrs.pop(param)
        # attrs['params'] = params
        attrs['numeric_params'] = numeric_params

        # Add fields/params and other instance attributes
        attrs['__slots__'] = (('_id', 'hid', *params, '__weakref__')
                              if not bases else ('_id', 'hid', *params))

        # Register the new class
        cls = super().__new__(mcs, typename, bases, attrs)

        # Create container for references to instances
        cls._instances = []

        return cls

    def __call__(cls, *args, **kwargs):  # call right before a new instance is created
        # register new instance
        instance = super().__call__(*args, **kwargs)

        # initialize fields/params
        for param in cls.__slots__[2:]:  # Exclude _id and __weakref__
            setattr(instance, param,
                    kwargs.get(param,
                               getattr(cls, param + '_type')()))
        # Set id
        instance._id = len(cls._instances)
        # Create ref
        cls._instances.append(weakref.ref(instance))
        # no default higher cluster id, set to None
        instance.hid = None  # higher cluster's id

        return instance

    def get_instance(cls, cluster_id):
        try:
            return cls._instances[cluster_id]()
        except IndexError:
            return None

    @property
    def instance_cnt(cls):
        return len(cls._instances)


# ----------------------------------------------------------------------------
# ClusterStructure class
class ClusterStructure(metaclass=MetaCluster):
    """
    Class for cluster objects in CogAlg.
    Each time a new instance is created, four things are done:
    - Set initialize field/param.
    - Set id.
    - Save a weak reference of instance inside the class object.
    (meaning that if there's no other references to instance,
    it will be garbage collected, weakref to it will return None
    afterwards)
    - Set higher cluster id to None (no higher cluster structure yet)
    Examples
    --------
    >>> from class_cluster import ClusterStructure
    >>> class CP(ClusterStructure):
    >>>     L = int  # field/param name and default type
    >>>     I = int
    >>>
    >>> P1 = CP(L=1, I=5) # initialized with values
    >>> print(P1)
    CP(L=1, I=5)
    >>> P2 = CP()  # default initialization
    >>> print(P2)
    CP(L=0, I=0)
    >>> print(P1.id, P2.id)  # instance's ids
    0 1
    >>> # look for object by instance's ids
    >>> print(CP.get_instance(0), CP.get_instance(1))
    CP(L=1, I=5) CP(L=0, I=0)
    >>> P2.L += 1; P2.I += 10  # assignment, fields are mutable
    >>> print(P2)
    CP(L=1, I=10)
    >>> # Accumulate using accumulate()
    >>> P1.accumulate(L=1, I=2)
    >>> print(P1)
    CP(L=2, I=7)
    >>> # ... or accum_from()
    >>> P2.accum_from(P1)
    >>> print(P2)
    CP(L=3, I=17)
    >>> # field/param types are not constrained, so be careful!
    >>> P2.L = 'something'
    >>> print(P2)
    CP(L='something', I=10)
    """

    def __init__(self, **kwargs):
        pass

    def accum_from(self, other, excluded=()):
        """Accumulate params from another structure."""
        self.accumulate(**{param: getattr(other, param, 0)
                           for param in self.numeric_params
                           if param not in excluded})


    def comp_param(self, other, ave, excluded=()):  # compare root layer to get 1st dm_layer

        # Get the subclass (inherited class) and init a new instance
        der = self.__class__.__subclasses__()[0]() # derCluster
        der.dm_layer = der.__class__.__subclasses__()[0]()

        # always exclude dy and dx related components
        if isinstance(excluded, str): # single element 'excluded' will be in string instead of tuple, since tuple need at least 2 elements
            excluded = (excluded, 'Dy', 'Dx', 'Day', 'Dax')
        else:
            excluded += ('Dy', 'Dx', 'Day', 'Dax')

        for param in self.numeric_params:
            if param not in excluded and param in other.numeric_params:
                p = getattr(self, param)
                _p = getattr(other, param)

                d = p - _p  # difference
                if param == 'I':
                    m = ave - abs(d)  # indirect match
                else:
                    m = min(p,_p) - abs(d)/2 - ave  # direct match
                dm = Cdm(d, m)  # dm instance

                # assign:
                setattr(der, param, p)            # set root param
                setattr(der.dm_layer, param, dm) # set dm in dm_layer

        if 'Dy' in self.numeric_params and 'Dy' in other.numeric_params:
            dy = getattr(self, 'Dy'); _dy = getattr(other, 'Dy')
            dx = getattr(self, 'Dx'); _dx = getattr(other, 'Dx')
            a =  dx + 1j * dy; _a = _dx + 1j * _dy # angle in complex form
            da = a * _a.conjugate()                # angle difference
            ma = ave - abs(da)                     # match
            setattr(der, 'Vector', a ) # set root param
            setattr(der.dm_layer, 'Vector', Cdm(da, ma)) # set dm in dm_layer

        if 'Day' in self.numeric_params and 'Day' in other.numeric_params:
            day = getattr(self, 'Day'); _day = getattr(other, 'Day')
            dax = getattr(self, 'Dax'); _dax = getattr(other, 'Dax')

            # temporary workaround until there is a better way to find angle difference between Day, Dax
            dday = day * _day.conjugate() # angle difference of complex day
            ddax = dax * _dax.conjugate() # angle difference of complex dax
            # formula for sum of angles, ~ angle_diff:
            # daz = (cos_1*cos_2 - sin_1*sin_2) + j*(cos_1*sin_2 + sin_1*cos_2)
            #     = (cos_1 + j*sin_1)*(cos_2 + j*sin_2)
            #     = az1 * az2
            dda = dday * ddax   # sum of angle difference
            mda = ave - abs(dda) # match
            setattr(der, 'aVector', (day, dax))  # set root param
            setattr(der.dm_layer, 'aVector', Cdm(dda, mda)) # set dm in dm_layer

        return der


    def comp_dm(self, other, ave, excluded=()):  # compare dm layer to get subsequent dm layer

        der = self.__class__.__subclasses__()[0]() # derCluster
        der.dm_layer = self.__class__.__subclasses__()[0]() # dm_layer has same class as derCluster

        for param in self.numeric_params:
            if param not in excluded and param in other.numeric_params:
                dmi = getattr(self, param)   # dm instance
                _dmi = getattr(other, param)

                if isinstance(dmi.d, complex):  # vector and avector
                    dd = dmi.d * _dmi.d.conjugate()  # angle difference of d
                    md = ave - abs(dd)  # match of d
                    dm = dmi.m - _dmi.m  # difference of m
                    mm = min(dmi.m, _dmi.m) - abs(dm) / 2 - ave  # match of m
                else:
                    dd = dmi.d - _dmi.d  # difference of d
                    md = min(dmi.d, _dmi.d) - abs(dd) / 2 - ave  # match of d
                    dm = dmi.m - _dmi.m  # difference of m
                    mm = min(dmi.m, _dmi.m) - abs(dm) / 2 - ave  # match of m

                d = Cdm(dd, md)  # difference and match of d
                m = Cdm(dm, mm)  # difference and match of m

                setattr(der, param, dmi)  # set root param
                setattr(der.dm_layer, param, Cdm(d, m)) # set dm in dm_layer

        return der



class Cdm(Number):
    __slots__ = ('d', 'm')

    def __init__(self, d=0, m=0):
        self.d, self.m = d, m

    def __add__(self, other):
        return Cdm(self.d + other.d, self.m + other.m)


    def __repr__(self):  # representation of object
        if isinstance(self.d, Cdm) or isinstance(self.m, Cdm):
            return "Cdm(d=Cdm, m=Cdm)"
        else:
            return "Cdm(d={}, m={})".format(self.d, self.m)


if __name__ == "__main__":  # for tests
    class CP(ClusterStructure):
        L = int
        D = int
        I = int
        M = int

    class CderP(CP):
        dm_layer = object
        mP=int

    class CPP(CderP):
        pass

    b = CP(L=1,D=2,I=5,M=9)
    c = CP(L=4,D=5,I=7,M=7)
    d = CP(L=8,D=6,I=2,M=1)
    derP1 = b.comp_param(c, ave=15)  # automatically return the inherited class (it is assumed to contain ders)
    #derP2 = c.comp_param(d, ave=15)
    #derP.L.accum()
    print(derP1.dm_layer)
    
    #print(derP2)
    #derP3 = derP1.comp_param_layer(derP2, ave=15, excluded=('mP')) 
    #print(derP3)
