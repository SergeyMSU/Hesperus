# Компилятор и флаги
CXX = nvcc
CXXFLAGS = -O3
LDFLAGS =

# Имя итогового исполняемого файла
TARGET = program

# Список объектных файлов
OBJS = kernel.o Konstruktor.o Kyb.o

# Цель по умолчанию
all: $(TARGET)

# Линковка объектных файлов в исполняемый
$(TARGET): $(OBJS)
	$(CXX) $(LDFLAGS) -o $@ $^

# Правило для компиляции .cu файлов
%.o: %.cu
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Правило для компиляции .cpp файлов
%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Очистка временных файлов
clean:
	rm -f $(OBJS) $(TARGET)

# Пересборка с нуля
rebuild: clean all