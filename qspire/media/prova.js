"use strict";
var __createBinding = (this && this.__createBinding) || (Object.create ? (function(o, m, k, k2) {
    if (k2 === undefined) k2 = k;
    var desc = Object.getOwnPropertyDescriptor(m, k);
    if (!desc || ("get" in desc ? !m.__esModule : desc.writable || desc.configurable)) {
      desc = { enumerable: true, get: function() { return m[k]; } };
    }
    Object.defineProperty(o, k2, desc);
}) : (function(o, m, k, k2) {
    if (k2 === undefined) k2 = k;
    o[k2] = m[k];
}));
var __setModuleDefault = (this && this.__setModuleDefault) || (Object.create ? (function(o, v) {
    Object.defineProperty(o, "default", { enumerable: true, value: v });
}) : function(o, v) {
    o["default"] = v;
});
var __importStar = (this && this.__importStar) || (function () {
    var ownKeys = function(o) {
        ownKeys = Object.getOwnPropertyNames || function (o) {
            var ar = [];
            for (var k in o) if (Object.prototype.hasOwnProperty.call(o, k)) ar[ar.length] = k;
            return ar;
        };
        return ownKeys(o);
    };
    return function (mod) {
        if (mod && mod.__esModule) return mod;
        var result = {};
        if (mod != null) for (var k = ownKeys(mod), i = 0; i < k.length; i++) if (k[i] !== "default") __createBinding(result, mod, k[i]);
        __setModuleDefault(result, mod);
        return result;
    };
})();
Object.defineProperty(exports, "__esModule", { value: true });
exports.activate = activate;

const vscode = __importStar(require("vscode"));
const fs = __importStar(require("fs"));
const path = __importStar(require("path"));

let panel;
















































































































// This version runs in the webview and uses message passing
window.callPythonExplain = function(filePath, smell, method) {
    return new Promise((resolve, reject) => {
        const path = require('path');
        const fs = require('fs');
        const { spawn } = require('child_process');
        
        // Get the project root the same way as your detection function
        const projectRoot = path.resolve(__dirname, '..'); // Go up one level from 'out' to project root
        
        // Determine Python command (same as your existing code)
        const pythonCmd = process.platform === 'win32' ? 'python' : 'python3';
        
        // Create temporary wrapper script path
        const wrapperScriptPath = path.join(projectRoot, 'temp_explain_wrapper.py');
        
        // Create the Python wrapper code
        const pythonCode = `import sys
import json
import os

# Add project root to Python path
sys.path.insert(0, r'${projectRoot}')

def serialize_explanation_result(obj):
    """Convert explanation result to JSON-serializable format"""
    if hasattr(obj, '__dict__'):
        # If it's a custom object, convert its attributes to a dictionary
        result = {}
        for key, value in obj.__dict__.items():
            if not key.startswith('_'):  # Skip private attributes
                try:
                    result[key] = serialize_explanation_result(value)
                except:
                    result[key] = str(value)
        return result
    elif isinstance(obj, list):
        return [serialize_explanation_result(item) for item in obj]
    elif isinstance(obj, dict):
        return {key: serialize_explanation_result(value) for key, value in obj.items()}
    elif isinstance(obj, (str, int, float, bool, type(None))):
        return obj
    else:
        return str(obj)

try:
    # Import the explain function from Explainer.py
    from Explainer import explain
    
    # Prepare the smell object - it might be a JSON string that needs parsing
    smell_data = """${JSON.stringify(smell).replace(/"/g, '\\"')}"""
    import json
    smell_obj = json.loads(smell_data)
    
    # Call the explain function
    result = explain(r'${filePath}', smell_obj, '${method}')
    
    # Serialize the result
    serialized_result = serialize_explanation_result(result)
    
    # Output the result
    if isinstance(serialized_result, str):
        # If it's already a string, wrap it in a success object
        print(json.dumps({"explanation": serialized_result, "success": True}))
    else:
        # If it's an object, serialize it
        print(json.dumps({"explanation": str(serialized_result), "success": True}))
        
except Exception as e:
    import traceback
    error_details = traceback.format_exc()
    error_result = {
        "error": str(e), 
        "details": error_details,
        "success": False
    }
    print(json.dumps(error_result))
`;

        // Write the wrapper script to a temporary file
        try {
            fs.writeFileSync(wrapperScriptPath, pythonCode, 'utf8');
        } catch (writeError) {
            console.error('Failed to create explain wrapper script:', writeError);
            reject(new Error(`Failed to create explain wrapper script: ${writeError.message}`));
            return;
        }

        const pythonProcess = spawn(pythonCmd, [wrapperScriptPath], {
            cwd: projectRoot,
            shell: true,
            stdio: ['pipe', 'pipe', 'pipe'],
            env: {
                ...process.env,
                PYTHONPATH: projectRoot + (process.env.PYTHONPATH ? path.delimiter + process.env.PYTHONPATH : '')
            }
        });
        
        let output = '';
        let errorOutput = '';
        
        pythonProcess.stdout.on('data', (data) => {
            const dataStr = data.toString();
            output += dataStr;
            console.log('Python explain stdout chunk:', dataStr);
        });
        
        pythonProcess.stderr.on('data', (data) => {
            const dataStr = data.toString();
            errorOutput += dataStr;
            console.error('Python explain stderr:', dataStr);
        });
        
        pythonProcess.on('close', (code) => {
            console.log(`Python explain process exited with code ${code}`);
            console.log('Full Python explain output:', output);
            console.log('Full Python explain error output:', errorOutput);
            
            // Clean up the temporary wrapper script
            try {
                if (fs.existsSync(wrapperScriptPath)) {
                    fs.unlinkSync(wrapperScriptPath);
                    console.log('Cleaned up temporary explain wrapper script');
                }
            } catch (cleanupError) {
                console.warn('Failed to cleanup explain wrapper script:', cleanupError);
            }
            
            if (code !== 0) {
                reject(new Error(`Python process exited with code ${code}. Error: ${errorOutput}`));
                return;
            }
            
            try {
                // Parse the JSON output from Python
                const result = JSON.parse(output.trim());
                
                if (result.success) {
                    resolve(result.explanation);
                } else {
                    reject(new Error(`Python error: ${result.error}\nDetails: ${result.details || ''}`));
                }
            } catch (parseError) {
                console.error('Failed to parse Python output:', output);
                reject(new Error(`Failed to parse Python output: ${parseError.message}\nOutput: ${output}`));
            }
        });
        
        pythonProcess.on('error', (error) => {
            console.error('Failed to start Python process:', error);
            reject(new Error(`Failed to start Python process: ${error.message}`));
        });
    });
}