# Python

### Features
* Friendly Syntax
* Free and Open Source
* Object-Oriented Language
* Dynamically Typed
* Interpreted
* Portable
<hr>

### A few important things to know
* To run Python programs, you will need the Python interpreter and possibly a text editor.
* A Python interpreter executes Python code (sometimes called programs).
* A program can be one or more Python files. Code files can include other files or modules. 

### Interpreted and Compiled Overview
* In the simple model of the world, “compile” means to convert a program in a high-level language into a binary executable full of machine code (CPU instructions). 
* When you compile a C program, this is what happens. The result is a file that your operating system can run for you.
* In the simple definition of “interpreted”, executing a program means reading the source code a line at a time, and doing what it says. This is the way some shells operate.
* In Python, the source code is compiled into a much simpler form called bytecode. 
* These are instructions similar in spirit to CPU instructions, but instead of being executed by the CPU, they are executed by software called a virtual machine. (These are not VM’s that emulate entire operating systems, just a simplified CPU execution environment.)
* Is Python compiled? Yes. Is Python interpreted? Yes.
#### Why Python is considered both:
**Compiled**: Because it converts source code to bytecode before execution.

**Interpreted**: Because the bytecode is executed by the PVM(Python Virtual Machine), which works as an interpreter.

* In terms of raw performance, Python is definitely slower than Java, C# and C/C++. However, there are other things that matter for the user/observer such as total memory usage, initial startup time, etc. For most things, Python is fast enough ;) . Check [this](https://benchmarksgame-team.pages.debian.net/benchmarksgame/fastest/python3-java.html) for benchmarks

<hr>

### Jupyter-Notebook
* While python code can be run in IDLE , and *.py* files we recommend you to download jupyter-notebook. Refer [this](https://www.geeksforgeeks.org/how-to-install-jupyter-notebook-in-windows/).
* If you don't have anaconda, refer to its official site to download.
* Read more [here](https://programminghistorian.org/en/lessons/jupyter-notebooks).

<hr>


### Applications of Python

Python is a versatile programming language widely used in various fields. Below are some of its key applications:

1. Web Development:
Python powers websites and APIs with frameworks like Django and Flask.

2. Data Science:
Python is used for data analysis, visualization, and preprocessing with libraries like Pandas and Matplotlib.

3. Machine Learning and Artificial Intelligence:
Python enables training and deploying AI models using TensorFlow, PyTorch, and scikit-learn.

4. Scientific Computing:
Python handles mathematical computations and simulations using NumPy and SciPy.

5. Automation:
Python automates repetitive tasks, web scraping, and testing workflows.

6. Game Development:
Python is used to develop 2D/3D games and prototypes with Pygame.

7. Cybersecurity:
Python is popular for penetration testing and developing security tools.

8. Finance:
Python supports algorithmic trading, financial analysis, and risk management.

9. Desktop Applications:
Python builds GUI-based desktop apps using Tkinter, PyQt, or Kivy.

10. IoT and Embedded Systems:
Python programs IoT devices and hardware using MicroPython.

11. Cloud Computing and DevOps:
Python automates cloud management and DevOps workflows with Boto3.

<hr>

# Table Of Contents
* [Basics Of Python](1_python_basics.ipynb)
    * [Data Types and Variables](1_python_basics.ipynb)
    * [Naming](1_python_basics.ipynb)
    * [Multiple Assignment](1_python_basics.ipynb)
    * [Python Numbers](1_python_basics.ipynb)
    * [Strings](1_python_basics.ipynb)
    * [String Slicing](1_python_basics.ipynb)
    * [Input](1_python_basics.ipynb)
    * [Python Lists](1_python_basics.ipynb)
        * [List Slicing](1_python_basics.ipynb)
    * [Python Tuples](1_python_basics.ipynb)
    * [Python Dictionary](1_python_basics.ipynb)
* [Operators And Control Statements](2_operators_and_conrol_statements.ipynb)
    * [Types Of Operations in python](2_operators_and_conrol_statements.ipynb)
        * [Arithmetic Operators](2_operators_and_conrol_statements.ipynb)
        * [Assignment Operator](2_operators_and_conrol_statements.ipynb)
        * [Comparision Operatosr](2_operators_and_conrol_statements.ipynb)
        * [Logical Operators](2_operators_and_conrol_statements.ipynb)
        * [Identity Operators](2_operators_and_conrol_statements.ipynb)
        * [Membership Operators](2_operators_and_conrol_statements.ipynb)
        * [Bitwise Operators](2_operators_and_conrol_statements.ipynb)
    * [Control Statements](2_operators_and_conrol_statements.ipynb)
        * [if elif else](2_operators_and_conrol_statements.ipynb)
        * [Looping Statements](2_operators_and_conrol_statements.ipynb)
            * [For loop](2_operators_and_conrol_statements.ipynb)
            * [While loop](2_operators_and_conrol_statements.ipynb)

* [Lists In Python](3_lists.ipynb)
    * [Python List Slicing](3_lists.ipynb)
        * [Slicing Using Positive Indexes](3_lists.ipynb) 
        * [Slicing Using Negative Indexes](3_lists.ipynb)
    * [Changing Items Of List](3_lists.ipynb)
        * [len](3_lists.ipynb)
        * [range](3_lists.ipynb)
    * [List Methods](3_lists.ipynb)
        * [Inserting Items](3_lists.ipynb)
        * [Append Items](3_lists.ipynb)
        * [Extend List](3_lists.ipynb)
        * [Remove Elements](3_lists.ipynb)
    * [Looping Through a List]([](3_lists.ipynb))
    * [List Comphrehension]([](3_lists.ipynb))
* [Dictionaries In Python](4_dictionary.ipynb)
    * [Accessing Items](4_dictionary.ipynb)
    * [Adding And Updating a Dictionary](4_dictionary.ipynb)
    * [Removing Items](4_dictionary.ipynb)
    * [Dictionary Key Method](4_dictionary.ipynb)
    * [Dictionary Values Method](4_dictionary.ipynb)
    * [Looping Through a Dictionary](4_dictionary.ipynb)
    * [Copying A Dictionary ](4_dictionary.ipynb)
    * [Nested Dictionaries](4_dictionary.ipynb)
* [Sets And Tuples](5_sets_and_tuples.ipynb)
    * [Set Datatype](5_sets_and_tuples.ipynb)
    * [Frozenset Datatype](5_sets_and_tuples.ipynb)
    * [Tuple Datatype](5_sets_and_tuples.ipynb)
* [Functions](6_functions.ipynb)
    * [Syntax Of Function](6_functions.ipynb)
    * [Aruguments](6_functions.ipynb)
    * [Number Of Arguments](6_functions.ipynb)
        * [Keyword Arguments](6_functions.ipynb)
        * [Arbitrary Arguments](6_functions.ipynb)
        * [Arbitrary Keyword Arguments](6_functions.ipynb)
        * [Default Parameter Value](6_functions.ipynb)
        * [Passing List as Argument](6_functions.ipynb)
    * [Return Values](6_functions.ipynb)
    * [Recursion](6_functions.ipynb)
    * [Pass By Reference or Pass By value](6_functions.ipynb)
    * [Docstrings](6_functions.ipynb)
    * [Lambda Function](6_functions.ipynb)
* [Classes And Objects](7_classes_and_objects.ipynb)
    * [Overview Of Classes](7_classes_and_objects.ipynb)
    * [Overview Of Methods](7_classes_and_objects.ipynb)
    * [General Format of Classes In Python](7_classes_and_objects.ipynb)
    * [Constructor](7_classes_and_objects.ipynb)
    * [The Self Variable](7_classes_and_objects.ipynb)
* [Advance Topics](8_advance.ipynb)
    * [Lambda Function](8_advance.ipynb)
    * [Modules and Packages](8_advance.ipynb)
    * [Classes](8_advance.ipynb)
        * [Instance Attributes](8_advance.ipynb)
        * [Class Attributes](8_advance.ipynb)
        * [Purpose of self in Python Classes](8_advance.ipynb)
        * [Instance Methods](8_advance.ipynb)
        * [Class Methods](8_advance.ipynb)
        * [Static Methods](8_advance.ipynb)
        * [Encapsulation](8_advance.ipynb)
        * [Inheritance](8_advance.ipynb)
        * [Polymorphism](8_advance.ipynb)
        * [Abstraction](8_advance.ipynb)
        * [Composition](8_advance.ipynb)
        * [Abstraction](8_advance.ipynb)
    * [dir function](8_advance.ipynb)