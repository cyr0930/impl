def raw_string():
    print(r'C:\some\name')


def concat_string_literals():
    """This feature is defined at the syntactical level, but implemented at compile time."""
    print('Usage: thingy [OPTIONS]\n'
          '-h                        Display this usage message\n'
          '-H hostname               Hostname to connect to\n')


def else_clauses_on_loops():
    """Executed when the loop terminates through exhaustion of the iterable (with for)
    or when the condition becomes false (with while),
    but not when the loop is terminated by a break statement.
    """
    for n in range(2, 10):
        for x in range(2, n):
            if n % x == 0:
                print(n, 'equals', x, '*', n//x)
                break
        else:
            # loop fell through without finding a factor
            print(n, 'is a prime number')


def default_argument_values_with_mutable_object(a, L=[]):
    """The default value is evaluated only once."""
    L.append(a)
    print(L)


def keyword_arguments(kind, *arguments, hr="-" * 40, **keywords):
    """The order in which the keyword arguments are printed is guaranteed to match the order in which they were provided
    in the function call.
    A final formal parameter of the form *name receives a tuple.
    A final formal parameter of the form **name receives a dictionary.
    Any formal parameters which occur after the *name parameter are ‘keyword-only’ arguments.
    *name must occur before **name.
    """
    print("-- Do you have any", kind, "?")
    print("-- I'm sorry, we're all out of", kind)
    for arg in arguments:
        print(arg)
    print(hr)
    for kw in keywords:
        print(kw, ":", keywords[kw])


def unpacking_argument_lists():
    def parrot(voltage, state='a stiff', action='voom'):
        print("-- This parrot wouldn't", action, end=' ')
        print("if you put", voltage, "volts through it.", end=' ')
        print("E's", state, "!")
    args = [3, 6]
    print(list(range(*args)))
    d = {"voltage": "four million", "state": "bleedin' demised", "action": "VOOM"}
    parrot(**d)


def function_annotations(ham: str, eggs: str = 'eggs') -> str:
    """Have no effect on any other part of the function."""
    print("Annotations:", function_annotations.__annotations__)
    print("Arguments:", ham, eggs)
    return ham + ' and ' + eggs


def list_comprehensions():
    """
    combs = []
    for x in [1,2,3]:
        for y in [3,1,4]:
            if x != y:
                combs.append((x, y))
    """
    print([(x, y) for x in [1,2,3] for y in [3,1,4] if x != y])


def del_statement():
    a = [-1, 1, 66.25, 333, 333, 1234.5]
    del a[0]
    print(a)
    del a[2:4]
    print(a)
    del a[:]
    print(a)
    del a
    # Referencing the name a hereafter is an error


def dict_constructor():
    """Builds dictionaries directly from sequences of key-value pairs."""
    print(dict([('sape', 4139), ('guido', 4127), ('jack', 4098)]))
    print(dict(sape=4139, guido=4127, jack=4098))


def zip_function():
    questions = ['name', 'quest', 'favorite color']
    answers = ['lancelot', 'the holy grail', 'blue']
    for q, a in zip(questions, answers):
        print('What is your {0}?  It is {1}.'.format(q, a))


def boolean_operator_not_as_a_boolean():
    string1, string2, string3 = '', 'Trondheim', 'Hammer Dance'
    print(string1 or string2 or string3)


def str_and_repr():
    """The str() function is meant to return representations of values which are fairly human-readable,
    while repr() is meant to generate representations which can be read by the interpreter
    (or will force a SyntaxError if there is no equivalent syntax)"""
    print(str('hello, world\n'))    # The repr() of a string adds string quotes and backslashes
    print(repr('hello, world\n'))


def formatted_string_literals():
    import math
    print(f'The value of pi is approximately {math.pi:.3f}.')
    table = {'Sjoerd': 4127, 'Jack': 4098, 'Dcab': 7678}
    for name, phone in table.items():
        print(f'{name:10} ==> {phone:10d}')
    animals = 'eels'
    print(f'My hovercraft is full of {animals}.')
    print(f'My hovercraft is full of {animals!a}.')     # ascii()
    print(f'My hovercraft is full of {animals!s}.')     # str()
    print(f'My hovercraft is full of {animals!r}.')     # repr()


def handling_exceptions(divisor):
    class Error(Exception):
        """Most exceptions are defined with names that end in “Error”,
        similar to the naming of the standard exceptions.
        """
        def __init__(self, expression, message):
            self.expression = expression
            self.message = message
    try:
        if divisor:
            x = 1/divisor
        else:
            raise ZeroDivisionError     # shorthand for 'raise ZeroDivisionError()'
    except ZeroDivisionError:
        try:
            try:
                raise Error('spam', 'eggs')
            except Error:
                raise
        except Error as inst:
            print(type(inst), inst.args, inst)
            print(inst.expression, inst.message)
    else:
        # It is useful for code that must be executed if the try clause does not raise an exception.
        # The use of the else clause is better than adding additional code to the try clause
        # because it avoids accidentally catching an exception that wasn’t raised by the code being protected
        # by the try … except statement.
        print(x)
        return 'Hello, world!'
    finally:
        # If an exception occurs during execution of the try clause, the exception may be handled by an except clause.
        # If the exception is not handled by an except clause, the exception is re-raised after the finally clause has
        # been executed.
        # An exception could occur during execution of an except or else clause. Again, the exception is re-raised after
        # the finally clause has been executed.
        # If the try statement reaches a break, continue or return statement, the finally clause will execute just prior
        # to the break, continue or return statement’s execution.
        # If a finally clause includes a return statement, the returned value will be the one from the finally clause’s
        # return statement, not the value from the try clause’s return statement.
        return 'Goodbye, world!'


def scopes_and_namespaces():
    def do_local():
        spam = "local spam"

    def do_nonlocal():
        nonlocal spam
        spam = "nonlocal spam"

    def do_global():
        global spam
        spam = "global spam"

    spam = "test spam"
    do_local()
    print("After local assignment:", spam)
    do_nonlocal()
    print("After nonlocal assignment:", spam)
    do_global()
    print("After global assignment:", spam)


def name_mangling():
    """Mangling rules are designed mostly to avoid accidents."""
    class Mapping:
        def __init__(self, iterable):
            self.items_list = []
            self.__update(iterable)

        def update(self, iterable):
            for item in iterable:
                self.items_list.append(item)
        __update = update  # private copy of original update() method

    class MappingSubclass(Mapping):
        def update(self, keys, values):
            # provides new signature for update(), but does not break __init__()
            for item in zip(keys, values):
                self.items_list.append(item)
    m = MappingSubclass([1, 2, 3])
    print(m._Mapping__update)
    print(m.items_list)


def iterators():
    class Reverse:
        def __init__(self, data):
            self.data = data
            self.index = len(data)

        def __iter__(self):
            # Define an __iter__() method which returns an object with a __next__() method.
            # If the class defines __next__(), then __iter__() can just return self.
            return self

        def __next__(self):
            if self.index == 0:
                raise StopIteration
            self.index = self.index - 1
            return self.data[self.index]
    rev = Reverse('spam')
    for char in rev:
        print(char)
    print(iter(rev))


def generators():
    """Generators are a simple and powerful tool for creating iterators."""
    def reverse(data):
        for index in range(len(data) - 1, -1, -1):
            yield data[index]
    for char in reverse('golf'):
        print(char)


def generator_expressions():
    """Generator expressions are more compact but less versatile than full generator definitions
    and tend to be more memory friendly than equivalent list comprehensions.
    """
    print(sum(i*i for i in range(10)))


def multi_threading():
    import threading

    class AsyncMsg(threading.Thread):
        def __init__(self, msg):
            threading.Thread.__init__(self)
            self.msg = msg

        def run(self):
            print('Finished background: ', self.msg)
    background = AsyncMsg('test')
    background.start()
    print('The main program continues to run in foreground.')
    background.join()  # Wait for the background task to finish
    print('Main program waited until background was done.')


def weak_references():
    import weakref, gc

    class A:
        def __init__(self, value):
            self.value = value

        def __repr__(self):
            return str(self.value)
    a = A(10)  # create a reference
    d = weakref.WeakValueDictionary()
    d['primary'] = a  # does not create a reference
    del a  # remove the one reference
    gc.collect()  # run garbage collection right away
    try:
        print(d['primary'])
    except Exception as e:
        print(type(e), e.args, e)


def decimal_floating_point_arithmetic():
    from decimal import Decimal
    print(round(Decimal('0.70') * Decimal('1.05'), 2))
    print(round(.70 * 1.05, 2))
    # Another form of exact arithmetic is supported by the fractions module
    # which implements arithmetic based on rational numbers (so the numbers like 1/3 can be represented exactly).


if __name__ == '__main__':
    import sys
    print('Start main at', sys.argv[0])
    print(dir(sys))
    raw_string()
    concat_string_literals()
    else_clauses_on_loops()
    default_argument_values_with_mutable_object(1)
    default_argument_values_with_mutable_object(2)
    keyword_arguments("Limburger",
                      "It's very runny, sir.", "It's really very, VERY runny, sir.",
                      hr="-" * 10,
                      shopkeeper="Michael Palin", client="John Cleese", sketch="Cheese Shop Sketch")
    unpacking_argument_lists()
    function_annotations('spam')
    list_comprehensions()
    del_statement()
    dict_constructor()
    zip_function()
    boolean_operator_not_as_a_boolean()
    str_and_repr()
    formatted_string_literals()
    print(handling_exceptions(0))
    print(handling_exceptions(10))
    scopes_and_namespaces()
    name_mangling()
    iterators()
    generators()
    generator_expressions()
    multi_threading()
    weak_references()
    decimal_floating_point_arithmetic()
