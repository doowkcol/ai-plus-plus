# AI++ - Complete Language Reference


AI++ adds powerful features that transform it into a complete programming language:

###  New Features

1. **Loops** - `while` and `for` loops for iteration
2. **Control Flow** - `break`, `continue`, and `return` statements
3. **File I/O** - Read/write text files and JSON
4. **Network/API** - HTTP GET/POST requests
5. **Multi-threading** - Parallel task execution
6. **GPU Acceleration** - CUDA support via CuPy
7. **Utility Functions** - `range`, `len`, and more

---

##  Table of Contents

1. [Loops](#loops)
2. [Control Flow](#control-flow)
3. [File I/O](#file-io)
4. [Network Operations](#network-operations)
5. [Parallel Execution](#parallel-execution)
6. [GPU Acceleration](#gpu-acceleration)
7. [Complete Examples](#complete-examples)

---

##  Loops

### For Loops

Iterate over ranges or lists.

**Syntax:**
```aipp
for (variable in iterable) {
    # loop body
}
```

**Examples:**

```aipp
# Range iteration
for (i in execute(range, 5)) {
    execute(print, i);  # Prints 0, 1, 2, 3, 4
}

# Range with start and end
for (i in execute(range, 5, 10)) {
    execute(print, i);  # Prints 5, 6, 7, 8, 9
}

# Range with step
for (i in execute(range, 0, 10, 2)) {
    execute(print, i);  # Prints 0, 2, 4, 6, 8
}

# List iteration
items = [1, 2, 3, 4, 5];
for (item in items) {
    squared = item * item;
    execute(print, "Square:", squared);
}

# Vector iteration
vectors = [<0.5, 0.8>, <0.3, 0.7>, <0.9, 0.2>];
for (vec in vectors) {
    execute(print, "Vector:", vec);
}
```

### While Loops

Execute code while a condition is true.

**Syntax:**
```aipp
while (condition) {
    # loop body
}
```

**Examples:**

```aipp
# Simple countdown
counter = 10;
while (counter > 0) {
    execute(print, counter);
    counter = counter - 1;
}

# Convergence loop
value = 50;
target = 100;
while (value < target) {
    value = execute(optimize, value);
    execute(print, "Current value:", value);
}

# Probabilistic loop
confidence = ~0.7;
while (confidence < 0.9) {
    confidence = execute(optimize, confidence);
    execute(print, "Confidence:", confidence);
}
```

### Nested Loops

Loops can be nested for multi-dimensional iteration.

```aipp
# 2D grid
for (i in execute(range, 3)) {
    for (j in execute(range, 3)) {
        value = i * 3 + j;
        execute(print, "Position [", i, ",", j, "] =", value);
    }
}

# Matrix operations
rows = 3;
cols = 3;
for (row in execute(range, rows)) {
    for (col in execute(range, cols)) {
        execute(print, "Cell:", row, col);
    }
}
```

---

##  Control Flow

### Break Statement

Exit a loop early.

**Syntax:**
```aipp
break;
```

**Example:**

```aipp
# Find first match
for (i in execute(range, 100)) {
    if (i > 50) {
        execute(print, "Found:", i);
        break;
    }
}

# Search with early exit
items = [5, 12, 3, 18, 7];
for (item in items) {
    if (item > 15) {
        execute(print, "Large item found:", item);
        break;
    }
}
```

### Continue Statement

Skip the rest of the current iteration.

**Syntax:**
```aipp
continue;
```

**Example:**

```aipp
# Process only odd numbers
for (i in execute(range, 10)) {
    if (i % 2 == 0) {
        continue;
    }
    execute(print, "Odd:", i);
}

# Filter and process
items = [1, -2, 3, -4, 5];
for (item in items) {
    if (item < 0) {
        continue;  # Skip negative
    }
    processed = execute(optimize, item);
    execute(print, "Processed:", processed);
}
```

### Return Statement

Exit a function early with a value.

**Syntax:**
```aipp
return value;
return;  # Return without value
```

**Example:**

```aipp
function find_first(items, threshold) {
    for (item in items) {
        if (item > threshold) {
            return item;
        }
    }
    return -1;  # Not found
}

result = execute(find_first, [5, 10, 15, 20], 12);
execute(print, "Result:", result);  # 15

function validate(value) {
    if (value < 0) {
        execute(print, "Invalid value");
        return;  # Early exit
    }
    execute(print, "Valid:", value);
}
```

---

##  File I/O

### Read File

Read text content from a file.

**Syntax:**
```aipp
content = execute(read_file, filepath);
```

**Example:**

```aipp
# Read a text file
data = execute(read_file, "data.txt");
if (data != null) {
    execute(print, "File content:", data);
} else {
    execute(print, "Error reading file");
}
```

### Write File

Write text content to a file.

**Syntax:**
```aipp
success = execute(write_file, filepath, content);
```

**Example:**

```aipp
# Write to a file
message = "Hello from AI++";
success = execute(write_file, "output.txt", message);
if (success) {
    execute(print, "File written successfully");
}

# Append data
data = "New line of data";
execute(write_file, "log.txt", data);
```

### Read JSON

Load JSON data from a file.

**Syntax:**
```aipp
data = execute(read_json, filepath);
```

**Example:**

```aipp
# Load configuration
config = execute(read_json, "config.json");
execute(print, "Config:", config);

# Access nested data
if (config != null) {
    model = config["model"];
    execute(print, "Model:", model);
}
```

### Write JSON

Save data as JSON to a file.

**Syntax:**
```aipp
success = execute(write_json, filepath, data);
```

**Example:**

```aipp
# Save configuration
config = {
    "model": "gpt-4",
    "temperature": 0.7,
    "max_tokens": 2048
};
execute(write_json, "config.json", config);

# Save results
results = {
    "accuracy": 0.95,
    "latency": 45.3,
    "predictions": [0.8, 0.9, 0.7]
};
execute(write_json, "results.json", results);
```

### File Processing Pipeline

```aipp
function process_files(filenames) {
    results = [];
    
    for (filename in filenames) {
        content = execute(read_file, filename);
        
        if (content != null) {
            processed = execute(optimize, content);
            output = "processed_" + filename;
            execute(write_file, output, processed);
            results = results + [output];
        }
    }
    
    return results;
}

files = ["data1.txt", "data2.txt", "data3.txt"];
outputs = execute(process_files, files);
execute(print, "Processed files:", outputs);
```

---

##  Network Operations

### HTTP GET

Make GET request to a URL.

**Syntax:**
```aipp
response = execute(http_get, url);
```

**Response Format:**
```python
{
    "status_code": 200,
    "text": "response body",
    "headers": {"Content-Type": "application/json", ...}
}
```

**Example:**

```aipp
# Simple GET request
response = execute(http_get, "https://api.github.com");
if (response != null) {
    execute(print, "Status:", response["status_code"]);
    execute(print, "Content:", response["text"]);
}

# API call
api_response = execute(http_get, "https://api.example.com/data");
if (api_response["status_code"] == 200) {
    execute(print, "Success!");
}
```

### HTTP POST

Make POST request with JSON data.

**Syntax:**
```aipp
response = execute(http_post, url, data);
```

**Example:**

```aipp
# Post JSON data
api_url = "https://api.example.com/predict";
payload = {
    "model": "gpt-4",
    "prompt": "Hello, AI!",
    "temperature": 0.7
};

response = execute(http_post, api_url, payload);
execute(print, "Response:", response["text"]);

# Send vector for inference
embedding = <0.5, 0.8, 0.3>;
data = {"embedding": embedding, "task": "classify"};
result = execute(http_post, "https://ml-api.com/infer", data);
```

### API Integration Example

```aipp
function call_ai_api(prompt) {
    url = "https://api.openai.com/v1/completions";
    data = {
        "model": "gpt-3.5-turbo",
        "prompt": prompt,
        "max_tokens": 100
    };
    
    response = execute(http_post, url, data);
    
    if (response["status_code"] == 200) {
        return response["text"];
    } else {
        execute(print, "API error:", response["status_code"]);
        return null;
    }
}

result = execute(call_ai_api, "Explain AI++");
execute(print, "API Response:", result);
```

---

##  Parallel Execution

Execute multiple tasks concurrently using threads.

**Syntax:**
```aipp
parallel {
    task1;
    task2;
    task3;
}
```

**Example:**

```aipp
# Simple parallel execution
parallel {
    execute(print, "Task A running");
    execute(print, "Task B running");
    execute(print, "Task C running");
}

# Parallel vector processing
parallel {
    v1 = execute(optimize, <0.5, 0.8, 0.3>);
    v2 = execute(optimize, <0.7, 0.2, 0.9>);
    v3 = execute(optimize, <0.4, 0.6, 0.7>);
}

# Parallel API calls
parallel {
    response1 = execute(http_get, "https://api1.com/data");
    response2 = execute(http_get, "https://api2.com/data");
    response3 = execute(http_get, "https://api3.com/data");
}
```

### Multi-Model Inference

```aipp
function parallel_inference(query) {
    results = [];
    
    parallel {
        model1_result = execute(optimize, query);
        model2_result = execute(optimize, query);
        model3_result = execute(optimize, query);
    }
    
    # Aggregate results
    consensus = (model1_result + model2_result + model3_result) / 3;
    return consensus;
}

query = <0.5, 0.7, 0.4>;
result = execute(parallel_inference, query);
```

---

##  GPU Acceleration

Accelerate vector operations using CUDA/GPU.

### Check GPU Availability

```aipp
has_gpu = execute(gpu_available);
if (has_gpu) {
    execute(print, "GPU acceleration available");
} else {
    execute(print, "Running on CPU");
}
```

### Convert to GPU

Move vector to GPU memory.

**Syntax:**
```aipp
gpu_vector = execute(to_gpu, cpu_vector);
```

**Example:**

```aipp
# Create vector on CPU
vec_cpu = <0.5, 0.8, 0.3, 0.2, 0.9>;
execute(print, "CPU Vector:", vec_cpu);

# Move to GPU
vec_gpu = execute(to_gpu, vec_cpu);
execute(print, "GPU Vector:", vec_gpu);

# Compute similarity on GPU
vec1_gpu = execute(to_gpu, <0.8, 0.3, 0.6>);
vec2_gpu = execute(to_gpu, <0.7, 0.4, 0.5>);
sim = execute(similarity, vec1_gpu, vec2_gpu);
execute(print, "GPU Similarity:", sim);
```

### Convert to CPU

Move vector from GPU to CPU memory.

**Syntax:**
```aipp
cpu_vector = execute(to_cpu, gpu_vector);
```

**Example:**

```aipp
# Process on GPU
vec_gpu = execute(to_gpu, <0.5, 0.8, 0.3>);
processed_gpu = execute(optimize, vec_gpu);

# Move back to CPU
result_cpu = execute(to_cpu, processed_gpu);
execute(print, "Result:", result_cpu);
```

### GPU Batch Processing

```aipp
function gpu_batch_process(vectors) {
    results = [];
    
    for (vec in vectors) {
        gpu_vec = execute(to_gpu, vec);
        processed = execute(optimize, gpu_vec);
        cpu_result = execute(to_cpu, processed);
        results = results + [cpu_result];
    }
    
    return results;
}

vectors = [
    <0.5, 0.8, 0.3>,
    <0.7, 0.2, 0.9>,
    <0.4, 0.6, 0.7>
];

results = execute(gpu_batch_process, vectors);
```

---

##  Complete Examples

### Example 1: Semantic Search Engine

```aipp
function build_search_index(documents) {
    index = [];
    idx = 0;
    
    for (doc in documents) {
        # Create embedding (simplified)
        embedding = <0.5, 0.8, 0.3>;
        entry = {"id": idx, "doc": doc, "vec": embedding};
        index = index + [entry];
        idx = idx + 1;
    }
    
    execute(write_json, "index.json", index);
    return index;
}

function search_index(query_vec, index, top_k) {
    scores = [];
    
    for (entry in index) {
        score = execute(similarity, query_vec, entry["vec"]);
        scores = scores + [{"entry": entry, "score": score}];
    }
    
    # Simple top-k (just return first k for demo)
    results = [];
    count = 0;
    for (item in scores) {
        if (count < top_k) {
            results = results + [item];
            count = count + 1;
        }
    }
    
    return results;
}

# Usage
docs = ["AI paper 1", "ML tutorial", "DL research"];
index = execute(build_search_index, docs);

query = <0.6, 0.7, 0.4>;
results = execute(search_index, query, index, 2);
execute(print, "Top results:", results);
```

### Example 2: Distributed ML Pipeline

```aipp
function ml_pipeline(data_files, model_configs) {
    results = [];
    
    # Load data
    datasets = [];
    for (file in data_files) {
        data = execute(read_json, file);
        if (data != null) {
            datasets = datasets + [data];
        }
    }
    
    # Train models in parallel
    parallel {
        for (config in model_configs) {
            model = execute(optimize, config);
            results = results + [model];
        }
    }
    
    # Ensemble predictions
    consensus = 0;
    for (model in results) {
        consensus = consensus + model;
    }
    consensus = consensus / execute(len, results);
    
    # Save results
    output = {"consensus": consensus, "models": results};
    execute(write_json, "pipeline_results.json", output);
    
    return consensus;
}
```

### Example 3: Real-Time Monitoring System

```aipp
function monitor_system(endpoints, interval, duration) {
    iteration = 0;
    max_iterations = duration / interval;
    
    while (iteration < max_iterations) {
        execute(print, "Monitoring iteration:", iteration);
        
        # Check all endpoints
        statuses = [];
        for (endpoint in endpoints) {
            response = execute(http_get, endpoint);
            if (response != null) {
                status = {"endpoint": endpoint, "code": response["status_code"]};
                statuses = statuses + [status];
            }
        }
        
        # Log status
        log_entry = {"iteration": iteration, "statuses": statuses};
        execute(write_json, "monitor_log.json", log_entry);
        
        # Alert if issues
        for (status in statuses) {
            if (status["code"] != 200) {
                execute(print, "ALERT: Issue with", status["endpoint"]);
            }
        }
        
        iteration = iteration + 1;
    }
}
```

### Example 4: Adaptive Learning System

```aipp
function adaptive_learning(training_data, epochs) {
    weights = <0.1, 0.1, 0.1>;
    best_weights = weights;
    best_accuracy = 0;
    
    for (epoch in execute(range, epochs)) {
        execute(print, "Epoch", epoch);
        
        # Train on batches
        for (batch in training_data) {
            weights = execute(optimize, weights);
        }
        
        # Evaluate
        metrics = execute(measure);
        accuracy = metrics["accuracy"];
        
        execute(print, "  Accuracy:", accuracy);
        
        # Save if improved
        if (accuracy > best_accuracy) {
            best_accuracy = accuracy;
            best_weights = weights;
            execute(print, "  New best!");
        }
        
        # Early stopping
        if (accuracy > 0.99) {
            execute(print, "Early stopping - target achieved");
            break;
        }
    }
    
    # Save final model
    model = {"weights": best_weights, "accuracy": best_accuracy};
    execute(write_json, "trained_model.json", model);
    
    return best_weights;
}
```

---

##  Installation

### Basic Installation

```bash
pip install numpy
```

### With Network Support

```bash
pip install numpy requests
```

### With GPU Support

```bash
pip install numpy cupy-cuda11x  # Replace with your CUDA version
```

### Complete Installation

```bash
pip install numpy requests cupy-cuda11x
```

---

##  Usage

### Command Line

```python
from aiplusplus import run_aiplusplus

code = '''
for (i in execute(range, 5)) {
    execute(print, i);
}
'''

run_aiplusplus(code)
```

### With GPU

```python
run_aiplusplus(code, use_gpu=True)
```

### Verbose Mode

```python
run_aiplusplus(code, verbose=True)
```

---

##  Summary

AI++ is now a complete programming language with:

-  **Loops** - for, while
-  **Control Flow** - break, continue, return
-  **File I/O** - read/write text and JSON
-  **Network** - HTTP GET/POST
-  **Threading** - parallel execution
-  **GPU** - CUDA acceleration
-  **Utilities** - range, len

**AI++ is ready for production AI systems!**

---

## Documentation Author

**Jack Lockwood** (Doowkcol)

*AI++ Programming Language - Complete Language Reference*
