# -*- coding: utf-8 -*-
"""python_record_part1.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1zaVBTyrMPUA905MSP2wRViWFX4LsqzMW

**1.1. Create a simple calculator in Python**
"""

num1 = int(input("Enter Number 1: "))
operator = input("Enter the Operator: ")
num2 = int(input("Enter Number 2: "))
if operator == '+': print(num1 + num2)
elif operator == '-': print(num1 - num2)
elif operator == '*': print(num1 * num2)
elif operator == '/' and num2 != 0:
               print(num1/num2)
else: print("Error")

"""1.2 An electric power distribution company charges domestic customers as follows: Consumption unit Rate of charge:
1.2.1. 0-200 Rs. 0.50 per unit
- 201-400 Rs. 0.65 per unit in excess of 200
- 401-600 Rs 0.80 per unit excess of 400
-	601 and above Rs 1.00per unit excess of 600
-	If the bill exceeds Rs. 400, then a surcharge of 15% will be charged, and the minimum bill should be Rs. 100/-
Create a Python program based on the scenario mentioned above.

"""

charge=0
unit=int(input("enter the unit"))
if unit <=200 :
  charge=unit*0.50
elif unit>200 and unit<=400: charge=(200*0.50)+((unit-200)*0.65)
elif unit>400 and unit<=600:
  sur_charge=((400*0.60)+((unit-400)*0.8))*1.5
  charge+=sur_charge
else:
  sur_charge=((600*0.80)+((unit-600)*1))*1.5
  charge+=sur_charge
if charge<100:
  print("bill is:100")
else:
  print("bill is:"+str(charge))

"""1.3 Print the pyramid of numbers using for loops"""

startRange = 16
endRange = 16

for i in range(1, 18):
    for j in range(1, 34):
        if j in range(startRange, endRange + 1):
            print("*", end="")
        else:
            print(" ", end="")

    startRange -= 1
    endRange += 1
    print()

"""1.4 Write a program to find the number and sum of all integers greater than 100 and less than 200 that are divisible by 7."""

sum=0
count=0
for num in range(100,200):
  if num%7==0:
    count+=1
    sum+=num
print("sum="+str(sum))

"""1.5 Write a recursive function to calculate the sum of numbers from 0 to 10"""

def sum(number):
    if number > 0:
        return number + sum(number - 1)
    return 0
n = int(input("Number: "))
print("Sum:", sum(n))

"""1.6 Write a Python program to reverse the digits of a given number and add them to the original. If the sum is not a palindrome, repeat this procedure."""

def reverseNum(num):
    rev = 0
    while num > 0:
                   rev *= 10
                   rev += (num % 10)
                   num = num // 10
    return rev
def isPalindrome(number):
         isPal = True
         if number != reverseNum(number):
                isPal = False
         return isPal

n = int(input("Enter Number: "))
n += reverseNum(n)
while(not isPalindrome(n)):
 n += reverseNum(n)
print(n)

"""1.7 Write a menu-driven program that performs the following operations on
strings
-	Check if the String is a Substring of Another String
-	Count Occurrences of Character
-	Replace a substring with another substring
-	Convert to Capital Letters

"""

import os

def checkSubstring():
    str1 = input("Enter String 1: ")
    str2 = input("Enter String 2: ")
    if str2 in str1:
        print(str2, "is a substring of", str1)
    else:
        print(str2, "is not a substring of", str1)
    input("Press Enter to continue...")

def countOccurrence():
    str_input = input("Enter a String: ")
    ch = input("Enter Character to check: ")
    count = str_input.count(ch)
    print(ch, "found", count, "times in", str_input)
    input("Press Enter to continue...")

def replaceSubstring():
    str1 = input("Enter String 1: ")
    str2 = input("Enter Substring: ")
    str3 = input("Enter Substring replacement: ")
    if str2 in str1:
        str1 = str1.replace(str2, str3)
        print("Updated String:", str1)
    else:
        print(str2, "not found in", str1)
    input("Press Enter to continue...")

def toCapital():
    str_input = input("Enter a String: ")
    print(str_input.upper())
    input("Press Enter to continue...")

def initMenu():
    choice = 0
    while choice != 5:
        os.system('cls' if os.name == 'nt' else 'clear')
        print("1. Check if the String is a Substring of Another String")
        print("2. Count Occurrences of Character")
        print("3. Replace a Substring with Another Substring")
        print("4. Convert to Capital Letters")
        print("5. Exit")

        choice = int(input("Enter Your Choice: "))
        if choice in range(1, 6):
            if choice == 1:
                checkSubstring()
            elif choice == 2:
                countOccurrence()
            elif choice == 3:
                replaceSubstring()
            elif choice == 4:
                toCapital()
        else:
            print("Enter a Valid Choice")
            input("Press Enter to continue...")

initMenu()

"""1.8 Write a function to find the factorial of a number but also store the factorials calculated in a dictionary."""

def factorial(n, memo={}):
    if n < 0:
        raise ValueError("Factorial is not defined for negative numbers.")
    if n == 0 or n == 1:
        return 1
    if n in memo:
        return memo[n]
    else:
        result = n * factorial(n - 1, memo)
        memo[n] = result
        return result

num = 5
print(f"The factorial of {num} is {factorial(num)}")
print(f"Memoized factorials: {factorial.__defaults__[0]}")

"""1.9 Perform various set operations
- 1.9.1 Set Union
- 1.9.2 Set Intersection
- 1.9.3 Set Difference

"""

set1 = {1, 2, 3, 4}
set2= {3, 4, 5, 6}
union_set = set1.union(set2)
print("Union using union() method:", union_set)
union_set_operator = set1 | set2
print("Union using | operator:", union_set_operator)
intersection_set = set1.intersection(set2)
print("Intersection using intersection() method:",intersection_set)
intersection_set_operator = set1 & set2
print("Intersection using & operator:", intersection_set_operator)
difference_set = set1.difference(set2)
print("Difference using difference() method (set1 - set2):", difference_set)
difference_set_operator = set1 - set2
print("Difference using - operator (set1 - set2):", difference_set_operator)

"""1.10 Create a dictionary to store the name, roll_no, and total_mark of N students. Now print the details of the student with the highest total_mark."""

students = [
{"name": "Alice", "roll_no": 101, "total_mark": 85},
{"name": "Bob", "roll_no": 102, "total_mark": 92},
{"name": "Charlie", "roll_no": 103, "total_mark": 78},
{"name": "David", "roll_no": 104, "total_mark": 95},
{"name": "Eve", "roll_no": 105, "total_mark": 88}
]
highest_mark_student = max(students, key=lambda student: student["total_mark"])
print("Details of the student with the highest total_mark:")
print(f"Name: {highest_mark_student['name']}")
print(f"Roll No: {highest_mark_student['roll_no']}")
print(f"Total Marks: {highest_mark_student['total_mark']}")

"""1.11 Write a Python program to copy the contents of a file into another file, line by line.

"""

def copy_file(source_file, destination_file):
    try:
        with open(source_file, 'r') as src:
            with open(destination_file, 'w') as dest:
                for line in src:
                    dest.write(line)
        print(f"Contents of '{source_file}' copied to '{destination_file}' successfully.")
    except IOError as e:
        print(f"Error copying file: {e}")

source_file = 'source.txt'
destination_file = 'destination.txt'
copy_file(source_file, destination_file)

"""1.12 Use the OS module to perform
- Create a directory
- Directory Listing
-	Search for “.py” files
-	Remove a particular file

"""

import os

def create_directory(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"Directory '{directory}' created successfully.")
        else:
            print(f"Directory '{directory}' already exists.")
    except OSError as e:
        print(f"Error creating directory '{directory}': {e}")

def list_directory(directory):
    try:
        files = os.listdir(directory)
        print(f"Listing of directory '{directory}':")
        for file in files:
            print(file)
    except OSError as e:
        print(f"Error listing directory '{directory}': {e}")

def search_files(directory, extension):
    try:
        print(f"Searching for files with extension '{extension}' in directory '{directory}':")
        for root, _, files in os.walk(directory):
            for file in files:
                if file.endswith(extension):
                    print(os.path.join(root, file))
    except OSError as e:
        print(f"Error searching files in directory '{directory}': {e}")

def remove_file(file_path):
    try:
        os.remove(file_path)
        print(f"File '{file_path}' removed successfully.")
    except OSError as e:
        print(f"Error removing file '{file_path}': {e}")

def create_test_file(file_path):
    try:
        with open(file_path, 'w') as f:
            f.write("This is a test file to be removed.\n")
        print(f"File '{file_path}' created successfully.")
    except OSError as e:
        print(f"Error creating file '{file_path}': {e}")

directory_name = "test_directory"
file_extension = ".py"
file_to_remove = "test_directory/file_to_remove.txt"

create_directory(directory_name)
list_directory(directory_name)
search_files(directory_name, file_extension)

create_test_file(file_to_remove)

if os.path.exists(file_to_remove):
    remove_file(file_to_remove)
else:
    print(f"File '{file_to_remove}' does not exist, so it cannot be removed.")

"""1.13 Create a simple banking application by using inheritance."""

class Account:
    def __init__(self, account_number, balance=0):
        self.account_number = account_number
        self.balance = balance

    def deposit(self, amount):
        self.balance += amount
        print(f"Deposited ${amount}. New balance is ${self.balance}.")

    def withdraw(self, amount):
        if self.balance >= amount:
            self.balance -= amount
            print(f"Withdrew ${amount}. New balance is ${self.balance}.")
        else:
            print("Insufficient funds.")

    def display_balance(self):
        print(f"Account Number: {self.account_number}")
        print(f"Current Balance: ${self.balance}")


class SavingsAccount(Account):
    def __init__(self, account_number, balance=0, interest_rate=0.01):
        super().__init__(account_number, balance)
        self.interest_rate = interest_rate

    def add_interest(self):
        interest = self.balance * self.interest_rate
        self.balance += interest
        print(f"Interest added. New balance is ${self.balance}.")


class CheckingAccount(Account):
    def __init__(self, account_number, balance=0, transaction_fee=1):
        super().__init__(account_number, balance)
        self.transaction_fee = transaction_fee

    def deduct_transaction_fee(self):
        self.balance -= self.transaction_fee
        print(f"Deducted transaction fee of ${self.transaction_fee}. New balance is ${self.balance}.")



savings_acc = SavingsAccount("SAV123", 1000, 0.02)
checking_acc = CheckingAccount("CHK456", 500)

savings_acc.display_balance()
savings_acc.deposit(500)
savings_acc.add_interest()
savings_acc.withdraw(200)

print()

checking_acc.display_balance()
checking_acc.deposit(300)
checking_acc.deduct_transaction_fee()
checking_acc.withdraw(100)