
# ax^2 + by + c = 0
import math

a = float(input("Nhập hệ số a: "))
b = float(input("Nhập hệ số b: "))
c = float(input("Nhập hệ số c: "))
print("Phương trình có dạng: ")
print(f"{a}x^2 + {b}x + {c} = 0")

delta = float
delta = b**2 - 4*a*c

print(f"Giá trị của delta: {delta}")

if delta < 0:
	print("Phương trình vô nghiệm")
elif delta == 0:
	x = -b/2*a
	print(f"Phương trình có nghiệm kép: x1 = x2 = {x}")
else:
	x1 = (-b + math.sqrt(delta)) / (2*a)
	x2 = (-b - math.sqrt(delta)) / (2*a)
	print(f"Phương trình có 2 nghiệm phân biệt:\n x1 = {x1} \n x2 = {x2}")