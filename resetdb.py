from database import VectorDatabase

db = VectorDatabase()

confirm = input("Tem certeza que deseja resetar o banco? (s/n): ")

if confirm.lower() == "s":
    db.reset_table()
    print("Banco resetado com sucesso!")
else:
    print("Cancelado.")