"use strict";

let globalPythonPath = undefined;

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

/*
function activate(context) {
    // Create status bar item
    const statusBarItem = vscode.window.createStatusBarItem(vscode.StatusBarAlignment.Left);
    statusBarItem.text = '$(graph) QSmell Tool';
    statusBarItem.command = 'mytool.toggleGui';
    statusBarItem.tooltip = 'Toggle QSmell Tool GUI';
    statusBarItem.show();
    context.subscriptions.push(statusBarItem);

    // Register the toggle command
    const toggleCommand = vscode.commands.registerCommand('mytool.toggleGui', () => {
        if (panel) {
            panel.dispose();
            panel = undefined;
        } else {
            createWebviewPanel(context);
        }
    });
    context.subscriptions.push(toggleCommand);

    // Register other commands for backward compatibility
    const analyzeCommand = vscode.commands.registerCommand('mytool.analyzeFile', () => {
        // This could open the GUI and start analysis directly
        if (!panel) {
            createWebviewPanel(context);
        }
    });
    context.subscriptions.push(analyzeCommand);

    const showGuiCommand = vscode.commands.registerCommand('mytool.showGui', () => {
        if (!panel) {
            createWebviewPanel(context);
        }
        panel.reveal();
    });
    context.subscriptions.push(showGuiCommand);





     // Get the absolute path to your extension
    const extensionPath = context.extensionPath;
    console.log('Extension path:', extensionPath);

    // Get the absolute path to the file you want to import
    const extensionJsPath = vscode.Uri.file(
        path.join(extensionPath, 'out', 'extension.js')
    );
    
    // Convert to webview URI
    const extensionJsUri = panel.webview.asWebviewUri(extensionJsPath);
    console.log('Extension.js webview URI:', extensionJsUri.toString());


}
*/

/*
function createWebviewPanel(context) {
    panel = vscode.window.createWebviewPanel(
        'qsmellTool',
        'QSmell Tool',
        vscode.ViewColumn.Beside, // Open beside the active editor
        {
            enableScripts: true,
            retainContextWhenHidden: true, // Keep the webview state when hidden
            localResourceRoots: [vscode.Uri.file(path.join(context.extensionPath, 'media'))]
        }
    );

    // Load the HTML content
    const htmlPath = path.join(context.extensionPath, 'media', 'gui.html');
    if (fs.existsSync(htmlPath)) {
        const html = fs.readFileSync(htmlPath, 'utf8');
        panel.webview.html = html;
    } else {
        // Fallback: use embedded HTML if gui.html doesn't exist
        panel.webview.html = getEmbeddedHTML();
    }

    // Handle disposal
    panel.onDidDispose(() => {
        panel = undefined;
    });

    // Handle messages from the webview
    panel.webview.onDidReceiveMessage(async (message) => {
        console.log('Extension received message:', message);
        
        switch (message.command) {
            case 'getActiveFile':
                console.log('Handling getActiveFile request');
                sendActiveFileInfo();
                break;
            
            case 'analyze':
                console.log('Handling analyze request');
                await handleAnalysis(message);
                break;
                
            default:
                console.log('Unknown message command:', message.command);
        }
    });

    // Send initial active file info after a short delay
    setTimeout(() => {
        console.log('Sending initial active file info');
        sendActiveFileInfo();
    }, 1000);

    // Set up listener for active editor changes
    const activeEditorListener = vscode.window.onDidChangeActiveTextEditor(() => {
        console.log('Active editor changed');
        if (panel) {
            sendActiveFileInfo();
        }
    });
    
    // Make sure to dispose the listener when panel is disposed
    panel.onDidDispose(() => {
        activeEditorListener.dispose();
    });
}
    */













async function activate(context) {

    // ✅ 1. Get Python extension and interpreter
    const pythonExt = vscode.extensions.getExtension("ms-python.python");
    let pythonPath = undefined;

    if (pythonExt) {
    if (!pythonExt.isActive) {
        await pythonExt.activate();
    }

    const pythonApi = pythonExt.exports;

    const execDetails = pythonApi.settings.getExecutionDetails();
    pythonPath = execDetails?.execCommand?.[0];

    console.log("🔍 Using Python interpreter:", pythonPath);
    } else {
    vscode.window.showWarningMessage("Python extension not found. Using system python.");
    }

    if (pythonExt) {
        await pythonExt.activate();
        const pythonApi = pythonExt.exports;
        const execDetails = pythonApi.settings.getExecutionDetails();
        globalPythonPath = execDetails?.execCommand?.[0];
        console.log("🔍 Using Python interpreter:", globalPythonPath);
    }

    // Register the toggle command
    const toggleCommand = vscode.commands.registerCommand('mytool.toggleGui', () => {
        console.log('=== TOGGLE GUI COMMAND ===');
        console.log('Active editor at button click:', vscode.window.activeTextEditor?.document.uri.fsPath);
        console.log('Visible editors:', vscode.window.visibleTextEditors.map(e => e.document.uri.fsPath));
        
        if (panel) {
            panel.dispose();
            panel = undefined;
        } else {
            createWebviewPanel(context);
        }
    });
    context.subscriptions.push(toggleCommand);

    // Register other commands for backward compatibility
    const analyzeCommand = vscode.commands.registerCommand('mytool.analyzeFile', () => {
        if (!panel) {
            createWebviewPanel(context);
        }
    });
    context.subscriptions.push(analyzeCommand);

    const showGuiCommand = vscode.commands.registerCommand('mytool.showGui', () => {
        if (!panel) {
            createWebviewPanel(context);
        }
        panel.reveal();
    });
    context.subscriptions.push(showGuiCommand);
   
}




function createWebviewPanel(context) {
    panel = vscode.window.createWebviewPanel(
        'qsmellTool',
        'QSpire',
        vscode.ViewColumn.Beside,
        {
            enableScripts: true,
            retainContextWhenHidden: true,
            localResourceRoots: [vscode.Uri.file(path.join(context.extensionPath, 'media'))]
        }
    );

    // ✅ Create webview URIs BEFORE loading HTML (so they can be injected)
    const extensionJsUri = panel.webview.asWebviewUri(
        vscode.Uri.file(path.join(context.extensionPath, 'out', 'extension.js'))
    );
    
    const utilsUri = panel.webview.asWebviewUri(
        vscode.Uri.file(path.join(context.extensionPath, 'utils', 'helpers.js'))
    );

    console.log('Extension.js webview URI:', extensionJsUri.toString());
    console.log('Helpers.js webview URI:', utilsUri.toString());

    // RIGHT AFTER creating the panel, before loading HTML, log this:
    console.log('=== PANEL CREATION DEBUG ===');
    console.log('All visible text editors:', vscode.window.visibleTextEditors.length);
    console.log('Active text editor:', vscode.window.activeTextEditor?.document.uri.fsPath);
    console.log('All visible editors paths:', vscode.window.visibleTextEditors.map(e => e.document.uri.fsPath));

    // Load the HTML Content
    const htmlPath = path.join(context.extensionPath, 'media', 'gui.html');
    console.log("html path is: ", htmlPath)

    if (fs.existsSync(htmlPath)) {
        let html = fs.readFileSync(htmlPath, 'utf8');
        
        // ✅ Inject the webview URIs, not local file paths
        html = html.replace('</head>', `
            <script>
                window.extensionJsUri = "${extensionJsUri.toString()}";
                window.utilsUri = "${utilsUri.toString()}";
            </script>
        </head>`);
        
        panel.webview.html = html;
    } else {
        panel.webview.html = getEmbeddedHTML();
    }






    // Send active file info after a small delay to ensure webview is ready
    // Then in your setTimeout:
    setTimeout(() => {
        console.log('=== TIMEOUT CHECK (after 100ms) ===');
        const activeEditor = vscode.window.activeTextEditor;
        console.log('Active editor exists:', !!activeEditor);
        
        if (activeEditor) {
            console.log('✅ Active file:', activeEditor.document.uri.fsPath);
            console.log('✅ Basename:', path.basename(activeEditor.document.uri.fsPath));
        } else {
            console.log('❌ No active editor!');
            console.log('Visible editors:', vscode.window.visibleTextEditors.length);
            
            // Try to get the first visible editor as fallback
            if (vscode.window.visibleTextEditors.length > 0) {
                const firstEditor = vscode.window.visibleTextEditors[0];
                console.log('📝 Using first visible editor:', firstEditor.document.uri.fsPath);
                
                if (panel) {
                    panel.webview.postMessage({
                        command: 'setActiveFile',
                        filePath: firstEditor.document.uri.fsPath,
                        fileName: path.basename(firstEditor.document.uri.fsPath)
                    });
                }
                return;
            }
        }
        
        if (activeEditor && panel) {
            panel.webview.postMessage({
                command: 'setActiveFile',
                filePath: activeEditor.document.uri.fsPath,
                fileName: path.basename(activeEditor.document.uri.fsPath)
            });
            console.log('✅ Message sent to webview');
        }
    }, 100);




    // Listen for active editor changes while the panel is open
    const changeEditorDisposable = vscode.window.onDidChangeActiveTextEditor(editor => {
        if (panel && editor) {
            panel.webview.postMessage({
                command: 'setActiveFile',
                filePath: editor.document.uri.fsPath,
                fileName: path.basename(editor.document.uri.fsPath)
            });
        }
    });

    // Handle disposal
    panel.onDidDispose(() => {
        panel = undefined;
        changeEditorDisposable.dispose();
    }, null, context.subscriptions);

    // Handle messages from the webview
    panel.webview.onDidReceiveMessage(async (message) => {
        // console.log('Extension received message:', message);
        
        switch (message.command) {
            case 'getActiveFile':
                // console.log('Handling getActiveFile request');
                const activeEditor = vscode.window.activeTextEditor;
                if (activeEditor && panel) {
                    panel.webview.postMessage({
                        command: 'setActiveFile',
                        filePath: activeEditor.document.uri.fsPath,
                        fileName: path.basename(activeEditor.document.uri.fsPath)
                    });
                }
                break;
            
            case 'analyze':
                console.log('Handling analyze request');
                await handleAnalysis(message);
                break;

            case 'callPythonExplain':
                try {
                    const result = await callPythonExplain(
                        message.filePath, 
                        message.smell, 
                        message.method
                    );
                    panel.webview.postMessage({
                        command: 'pythonExplainResult',
                        data: result
                    });
                } catch (error) {
                    panel.webview.postMessage({
                        command: 'pythonExplainError',
                        error: error.message
                    });
                }
                break;

            case 'getExtensionPath':
                panel.webview.postMessage({
                    command: 'extensionPath',
                    path: context.extensionPath,
                    extensionJsUri: extensionJsUri.toString()
                });
                break;
            

            case 'loadConfig':
                loadConfigFromFile(message.field, panel.webview);
                break;
                
            case 'saveConfig':
                saveConfigToFile(message.field, message.value, panel.webview);
                break;

                
            default:
                console.log('Unknown message command:', message.command);
        }
    });
}


// Function to load config from JSON file
/*
function loadConfigFromFile(field, webview) {
    try {

        // Get the path to your config.json
        const configPath = path.join(__dirname, '..', 'config.json');
        
        // Read the file
        const configData = fs.readFileSync(configPath, 'utf8');
        const config = JSON.parse(configData);

        console.log("Config")
        console.log(config)

        if (field=="CG" || field=="LPQ" || field=="IQ" || field=="IdQ" || field=="ROC" || field=="LC" || field=="NC" || field=="IM" )
        
        // Send the value back to the webview
        webview.postMessage({
            command: 'configLoaded',
            field: field,
            value: config[field] || ''
        });
        
        console.log(`Loaded ${field}:`, config[field]);
    } catch (error) {
        console.error('Error loading config:', error);
        webview.postMessage({
            command: 'configLoaded',
            field: field,
            value: ''
        });
    }
}
*/


// Function to load config from JSON file
function loadConfigFromFile(field, webview) {
    try {
        // Get the path to your config.json
        const configPath = path.join(__dirname, '..', 'config.json');
        
        // Read the file
        const configData = fs.readFileSync(configPath, 'utf8');
        const config = JSON.parse(configData);

        console.log("Config");
        console.log(config);

        // Valid smell fields
        const validFields = ["CG", "LPQ", "IQ", "IdQ", "ROC", "NC", "IM", "LC"];

        // 🟢 Handle LC special sub-fields first
        if (field === "LC-Threshold" || field === "LC-GateError") {
            const lcParam = field === "LC-Threshold" ? "threshold" : "gate_error";
            let value = '';

            if (config.Smells && config.Smells.LC) {
                const lcSmell = config.Smells.LC;

                // Try custom values first
                if (lcSmell.Detector?.custom_values?.[lcParam] !== undefined) {
                    value = lcSmell.Detector.custom_values[lcParam];
                    console.log(`Found custom LC value for ${lcParam}:`, value);
                }
                // Fallback to default
                else if (lcSmell.Detector?.default_values?.[lcParam] !== undefined) {
                    value = lcSmell.Detector.default_values[lcParam];
                    console.log(`Using default LC value for ${lcParam}:`, value);
                }
            }

            // Send back to webview
            webview.postMessage({
                command: 'configLoaded',
                field: field,
                parameter: lcParam,
                value: value
            });

            console.log(`Loaded ${field}:`, value);
            return; // ✅ stop here, since we handled it
        }
        

        if( !validFields.includes(field) ){
            // Send the value back to the webview
            webview.postMessage({
                command: 'configLoaded',
                field: field,
                value: config[field] || ''
            });
        }
        
        if (validFields.includes(field)) {
            let value = '';
            let parameterName = '';
            
            // Check if the smell exists in config
            if (config.Smells && config.Smells[field]) {
                const smell = config.Smells[field];
                
                // Check for custom values first
                if (smell.Detector && smell.Detector.custom_values) {
                    const customValues = smell.Detector.custom_values;
                    // Get the first key-value pair from custom_values
                    const customKeys = Object.keys(customValues);
                    if (customKeys.length > 0) {
                        parameterName = customKeys[0];
                        value = customValues[parameterName];
                        console.log(`Found custom value for ${field}.${parameterName}:`, value);
                    }
                }
                
                // If no custom value, fall back to default_values
                if (value === '' && smell.Detector && smell.Detector.default_values) {
                    const defaultValues = smell.Detector.default_values;
                    const defaultKeys = Object.keys(defaultValues);
                    if (defaultKeys.length > 0) {
                        parameterName = defaultKeys[0];
                        value = defaultValues[parameterName];
                        console.log(`Using default value for ${field}.${parameterName}:`, value);
                    }
                }
            }
            
            // Send the value back to the webview
            webview.postMessage({
                command: 'configLoaded',
                field: field,
                parameter: parameterName,
                value: value
            });
            
            console.log(`Loaded ${field}:`, value);
        } else {
            console.error(`Invalid field: ${field}`);
            webview.postMessage({
                command: 'configLoaded',
                field: field,
                parameter: '',
                value: ''
            });
        }
        
    } catch (error) {
        console.error('Error loading config:', error);
        webview.postMessage({
            command: 'configLoaded',
            field: field,
            parameter: '',
            value: ''
        });
    }
}




// Function to save config to JSON file
/*
function saveConfigToFile(field, value, webview) {
    try {
        const configPath = path.join(__dirname, '..', 'config.json');
        
        
        // Read existing config
        let config = {};
        if (fs.existsSync(configPath)) {
            const configData = fs.readFileSync(configPath, 'utf8');
            config = JSON.parse(configData);
        }
        
        // Update the field
        config[field] = value;
        
        // Write back to file
        fs.writeFileSync(configPath, JSON.stringify(config, null, 2), 'utf8');
        
        // Send success message back to webview
        webview.postMessage({
            command: 'configSaved',
            field: field,
            success: true
        });
        
        console.log(`Saved ${field}:`, value);
    } catch (error) {
        console.error('Error saving config:', error);
        webview.postMessage({
            command: 'configSaved',
            field: field,
            success: false
        });
    }
}*/

// Function to save config to JSON file
/*
function saveConfigToFile(field, value, webview) {
    try {
        const configPath = path.join(__dirname, '..', 'config.json');
        
        // Read existing config
        let config = {};
        if (fs.existsSync(configPath)) {
            const configData = fs.readFileSync(configPath, 'utf8');
            config = JSON.parse(configData);
        }
        
        // Valid smell fields
        const validFields = ["CG", "LPQ", "IQ", "IdQ", "ROC", "LC", "NC", "IM"];

        
        if (validFields.includes(field)) {
            // Make sure the Smells structure exists
            if (!config.Smells) {
                config.Smells = {};
            }
            
            // Check if the smell exists in config
            if (config.Smells[field]) {
                const smell = config.Smells[field];
                
                // Get the parameter name from default_values
                let parameterName = '';
                if (smell.Detector && smell.Detector.default_values) {
                    const defaultKeys = Object.keys(smell.Detector.default_values);
                    if (defaultKeys.length > 0) {
                        parameterName = defaultKeys[0];
                    }
                }
                
                // Initialize custom_values if it doesn't exist
                if (!smell.Detector.custom_values) {
                    smell.Detector.custom_values = {};
                }
                
                // Save to custom_values
                if (parameterName) {
                    smell.Detector.custom_values[parameterName] = value;
                    console.log(`Saved custom value for ${field}.${parameterName}:`, value);
                }
            } else {
                console.error(`Smell ${field} not found in config`);
                webview.postMessage({
                    command: 'configSaved',
                    field: field,
                    success: false,
                    error: 'Smell not found in config'
                });
                return;
            }
        } else {
            // For non-smell fields (like API_KEY, LLM_model), save directly
            config[field] = value;
        }
        
        // Write back to file with pretty formatting
        fs.writeFileSync(configPath, JSON.stringify(config, null, 2), 'utf8');
        
        // Send success message back to webview
        webview.postMessage({
            command: 'configSaved',
            field: field,
            success: true
        });
        
        console.log(`Saved ${field}:`, value);
    } catch (error) {
        console.error('Error saving config:', error);
        webview.postMessage({
            command: 'configSaved',
            field: field,
            success: false,
            error: error.message
        });
    }
}
*/

function saveConfigToFile(field, value, webview) {
    try {
        const configPath = path.join(__dirname, '..', 'config.json');
        
        // Read existing config
        let config = {};
        if (fs.existsSync(configPath)) {
            const configData = fs.readFileSync(configPath, 'utf8');
            config = JSON.parse(configData);
        }

        // Valid smell fields
        const validFields = ["CG", "LPQ", "IQ", "IdQ", "ROC", "LC", "NC", "IM"];

        // 🟢 Handle LC special sub-fields first
        if (field === "LC-Threshold" || field === "LC-GateError") {
            const lcParam = field === "LC-Threshold" ? "threshold" : "gate_error";

            // Ensure LC structure exists
            if (!config.Smells) config.Smells = {};
            if (!config.Smells.LC) config.Smells.LC = {};
            if (!config.Smells.LC.Detector) config.Smells.LC.Detector = {};
            if (!config.Smells.LC.Detector.custom_values) config.Smells.LC.Detector.custom_values = {};

            // Save the value
            config.Smells.LC.Detector.custom_values[lcParam] = value;
            console.log(`Saved LC custom value for ${lcParam}:`, value);

            // Write back to file
            fs.writeFileSync(configPath, JSON.stringify(config, null, 2), 'utf8');

            // Send success message back
            webview.postMessage({
                command: 'configSaved',
                field: field,
                success: true
            });

            return; // ✅ Stop here, special case handled
        }

        // 🟡 Normal smell saving
        if (validFields.includes(field)) {
            if (!config.Smells) config.Smells = {};

            if (config.Smells[field]) {
                const smell = config.Smells[field];

                // Get parameter name from default_values
                let parameterName = '';
                if (smell.Detector?.default_values) {
                    const defaultKeys = Object.keys(smell.Detector.default_values);
                    if (defaultKeys.length > 0) {
                        parameterName = defaultKeys[0];
                    }
                }

                // Ensure custom_values exists
                if (!smell.Detector.custom_values) {
                    smell.Detector.custom_values = {};
                }

                // Save to custom_values
                if (parameterName) {
                    smell.Detector.custom_values[parameterName] = value;
                    console.log(`Saved custom value for ${field}.${parameterName}:`, value);
                }
            } else {
                console.error(`Smell ${field} not found in config`);
                webview.postMessage({
                    command: 'configSaved',
                    field: field,
                    success: false,
                    error: 'Smell not found in config'
                });
                return;
            }
        } else {
            // For non-smell fields (e.g. API_KEY, LLM_model)
            config[field] = value;
        }

        // Write back to file
        fs.writeFileSync(configPath, JSON.stringify(config, null, 2), 'utf8');

        // Notify success
        webview.postMessage({
            command: 'configSaved',
            field: field,
            success: true
        });

        console.log(`Saved ${field}:`, value);
    } catch (error) {
        console.error('Error saving config:', error);
        webview.postMessage({
            command: 'configSaved',
            field: field,
            success: false,
            error: error.message
        });
    }
}






// Your Python explain handler
async function handlePythonExplain(data) {
    // Implement your Python integration here
    console.log('Python explain called with:', data);
    return { success: true, result: 'Analysis completed' };
}



function sendActiveFileInfo() {
    if (!panel) {
        console.log('Panel not available, cannot send file info');
        return;
    }

    const activeEditor = vscode.window.activeTextEditor;
    const visibleEditors = vscode.window.visibleTextEditors;
    
    /*
    console.log('Sending active file info:', {
        activeEditor: !!activeEditor,
        visibleEditorsCount: visibleEditors.length,
        activeFileName: activeEditor ? path.basename(activeEditor.document.fileName) : null,
        activeFilePath: activeEditor ? activeEditor.document.uri.fsPath : null
    });*/
    
    if (!activeEditor) {
        // No active editor
        panel.webview.postMessage({
            command: 'activeFileInfo',
            fileName: null,
            filePath: null,
            multipleFiles: visibleEditors.length > 1
        });
        // console.log('No active editor found');
    } else {
        // We have an active editor
        const fileName = path.basename(activeEditor.document.fileName);
        const filePath = activeEditor.document.uri.fsPath;
        
        panel.webview.postMessage({
            command: 'activeFileInfo',
            fileName: fileName,
            filePath: filePath,
            multipleFiles: false
        });
        // console.log('Sent active file info:', { fileName, filePath });
    }
}

async function handleAnalysis(message) {
    if (!panel) return;


        try {
        let filePath = null;
        let content = null;
        let fileName = null;

        console.log('handleAnalysis called with message:', message);

        if (message.useActiveFile) {
            // First, try to use the fileName from the message (sent from GUI)
            if (message.fileName) {
                filePath = message.fileName;
                fileName = path.basename(filePath);
                console.log('Using file path from GUI:', filePath);
                
                try {
                    content = fs.readFileSync(filePath, 'utf8');
                    console.log('File content loaded, length:', content.length);
                } catch (readError) {
                    console.error('Failed to read file:', readError);
                    throw new Error(`Failed to read file: ${readError.message}`);
                }
            } else {
                // Fallback: try active editor
                const activeEditor = vscode.window.activeTextEditor;
                console.log('No fileName in message, checking active editor:', {
                    activeEditor: !!activeEditor,
                    document: activeEditor ? !!activeEditor.document : false,
                    uri: activeEditor?.document?.uri?.fsPath
                });
                
                if (activeEditor && activeEditor.document) {
                    filePath = activeEditor.document.uri.fsPath;
                    content = activeEditor.document.getText();
                    fileName = path.basename(filePath);
                    console.log('Using active file:', { filePath, fileName, contentLength: content.length });
                } else {
                    console.error('No active file available and no fileName provided');
                    throw new Error('No file selected - please select a file first');
                }
            }
        } else if (message.fileName) {
            // Use the file path provided in the message
            filePath = message.fileName;
            fileName = path.basename(filePath);
            console.log('Using provided file path:', filePath);
            
            try {
                content = fs.readFileSync(filePath, 'utf8');
                console.log('File content loaded, length:', content.length);
            } catch (readError) {
                console.error('Failed to read file:', readError);
                throw new Error(`Failed to read file: ${readError.message}`);
            }
        } else {
            // Ask user to select a file
            console.log('Asking user to select a file');
            const uri = await vscode.window.showOpenDialog({
                canSelectMany: false,
                openLabel: 'Choose file to analyze',
                filters: {
                    'Python files': ['py'],
                    'Quantum files': ['qasm', 'q'],
                    'All files': ['*']
                }
            });

            if (!uri || uri.length === 0) {
                vscode.window.showWarningMessage('No file selected for analysis.');
                return;
            }

            filePath = uri[0].fsPath;
            content = fs.readFileSync(filePath, 'utf8');
            fileName = path.basename(filePath);
        }

        // Final validation
        if (!filePath || !fileName) {
            console.error('File path or name is missing:', { filePath, fileName });
            throw new Error('File path or name is missing');
        }

        console.log('Starting analysis with:', { filePath, fileName, method: message.method });




        // Final validation
        if (!filePath || !fileName) {
            console.error('File path or name is missing:', { filePath, fileName });
            throw new Error('File path or name is missing');
        }

        console.log('Starting analysis with:', { filePath, fileName, method: message.method });

        // Notify webview that analysis started
        panel.webview.postMessage({
            command: 'analysisStarted',
            fileName: fileName,
            method: message.method
        });

        // Show info message to user
        vscode.window.showInformationMessage(`Starting ${message.method} analysis on: ${fileName}`);

        // Call your Python analysis function
        const results = await performAnalysis(filePath, content, message.method);

        console.log('Analysis completed successfully:', results);

        // Send results back to webview
        panel.webview.postMessage({
            command: 'analysisComplete',
            results: results,
            fileName: fileName,
            method: message.method
        });

        vscode.window.showInformationMessage(`Analysis complete for: ${fileName}`);

    } catch (error) {
        console.error('Analysis error:', error);
        
        // Send error back to webview
        panel.webview.postMessage({
            command: 'analysisError',
            error: error.message
        });

        vscode.window.showErrorMessage(`Analysis failed: ${error.message}`);
    }
}

async function performAnalysis(filePath, content, method) {
    console.log("Siamo qui")
  let smells= await callPythonDetectSmells(filePath, method);
  console.log("Siamo qui smells")
  console.log(smells)
  return smells
}

async function callPythonDetectSmells(filePath, method) {
    const { spawn, execSync } = require('child_process');
    const path = require('path');
    const fs = require('fs');



    // Function to find available Python command
    async function findPythonCommand() {
        // Try to get Python from VS Code's Python extension
        try {
            const pythonExtension = vscode.extensions.getExtension('ms-python.python');
            if (pythonExtension) {
                if (!pythonExtension.isActive) {
                    await pythonExtension.activate();
                }
                const pythonPath = pythonExtension.exports.settings.getExecutionDetails?.()?.execCommand?.[0];
                if (pythonPath) {
                    console.log('Using Python from VS Code extension:', pythonPath);
                    return pythonPath;
                }
            }
        } catch (err) {
            console.log('Could not get Python from VS Code extension:', err);
        }
        
        // Fallback to trying common commands
        const commands = ['py', 'python3', 'python'];
        for (const cmd of commands) {
            try {
                execSync(`${cmd} --version`, { stdio: 'ignore', shell: true });
                return cmd;
            } catch (error) {
                continue;
            }
        }
        
        throw new Error('Python not found. Please install the Python extension for VS Code or add Python to your system PATH.');
    }

    const projectRoot = path.resolve(__dirname, '..'); // Go up one level from 'out' to project root
        
        let pythonScriptPath = "";

        if(method === 'dynamic'){ pythonScriptPath = path.join(projectRoot, 'detection', 'DynamicDetection', 'GeneralFileTest.py'); }
        else{ pythonScriptPath = path.join(projectRoot, 'detection', 'StaticDetection', 'StaticMappedDetection.py'); }
        
        //const pythonCmd = 'python';

        
        // Find the correct Python command instead of hardcoding
        /*
        let pythonCmd;
        try {
            pythonCmd = await findPythonCommand();
        } catch (error) {
            reject(error);
            return;
        }*/

        // Always prefer the interpreter VS Code is using (venv)
        let pythonCmd = globalPythonPath;

        if (!pythonCmd) {
            console.warn("⚠️ VS Code Python extension not available — fallback to system Python.");
            pythonCmd = await findPythonCommand();
        }
        console.log("pythonCmd: ",pythonCmd)

        console.log(`Project root: ${projectRoot}`);
        console.log(`Calling Python script: ${pythonScriptPath} with file: ${filePath}`);
        console.log(`Using Python command: ${pythonCmd}`);

        // Check if the Python file exists before trying to run it
        if (!fs.existsSync(pythonScriptPath)) {
            const errorMsg = `Python script not found at: ${pythonScriptPath}`;
            console.error(errorMsg);
            throw new Error(errorMsg);
        } else {
            console.log('Python script found at:', pythonScriptPath);
        }

    
    return new Promise((resolve, reject) => {
        // Get the correct path to your Python script
        









        
        // Check if the Python file exists before trying to run it
        const fs = require('fs');
        if (!fs.existsSync(pythonScriptPath)) {
            const errorMsg = `Python script not found at: ${pythonScriptPath}`;
            console.error(errorMsg);
            reject(new Error(errorMsg));
            return;
        } else {
            console.log('Python script found at:', pythonScriptPath);
        }
        
        // Create a wrapper script approach
        const wrapperScriptPath = path.join(projectRoot, 'temp_wrapper.py');
        
        // Create the Python wrapper code with serialization
        const pythonCode = `import sys
import json
import os

# Add project root to Python path
sys.path.insert(0, r'${projectRoot}')

def serialize_smell_objects(obj):
    """Convert smell objects to JSON-serializable dictionaries"""
    if hasattr(obj, '__dict__'):
        # If it's a custom object, convert its attributes to a dictionary
        result = {}
        for key, value in obj.__dict__.items():
            if not key.startswith('_'):  # Skip private attributes
                try:
                    # Recursively serialize nested objects
                    result[key] = serialize_smell_objects(value)
                except:
                    # If serialization fails, convert to string
                    result[key] = str(value)
        return result
    elif isinstance(obj, list):
        # Handle lists recursively
        return [serialize_smell_objects(item) for item in obj]
    elif isinstance(obj, dict):
        # Handle dictionaries recursively
        return {key: serialize_smell_objects(value) for key, value in obj.items()}
    elif isinstance(obj, (str, int, float, bool, type(None))):
        # Basic types that are JSON serializable
        return obj
    else:
        # For any other type, convert to string
        return str(obj)

try:
    # Import the appropriate module
    if '${method}' == 'dynamic':
        from detection.DynamicDetection.GeneralFileTest import detect_smells_from_file
        results = detect_smells_from_file(r'${filePath}')
    else:
        from detection.StaticDetection.StaticMappedDetection import detect_smells_from_static_file_forJS
        results = detect_smells_from_static_file_forJS(r'${filePath}')
    
    
    # Serialize the results to make them JSON-compatible
    serialized_results = serialize_smell_objects(results)
    
    # Output results as JSON
    if isinstance(serialized_results, dict):
        print(json.dumps(serialized_results))
    elif isinstance(serialized_results, list):
        # If it's a list of smell dictionaries, wrap it properly
        print(json.dumps({"smells": serialized_results, "success": True, "count": len(serialized_results)}))
    else:
        print(json.dumps({"smells": [serialized_results], "success": True}))
        
except Exception as e:
    error_result = {"error": str(e), "smells": [], "success": False}
    print(json.dumps(error_result))
`;

        // Write the wrapper script to a temporary file
        try {
            fs.writeFileSync(wrapperScriptPath, pythonCode, 'utf8');
        } catch (writeError) {
            console.error('Failed to create wrapper script:', writeError);
            reject(new Error(`Failed to create wrapper script: ${writeError.message}`));
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
            console.log('Python stdout chunk:', dataStr);
        });
        
        pythonProcess.stderr.on('data', (data) => {
            const dataStr = data.toString();
            errorOutput += dataStr;
            console.error('Python stderr:', dataStr);
        });
        
        pythonProcess.on('close', (code) => {
            console.log(`Python process exited with code ${code}`);
            console.log('Full Python output:', output);
            console.log('Full Python error output:', errorOutput);
            
            // Clean up the temporary wrapper script
            try {
                if (fs.existsSync(wrapperScriptPath)) {
                    fs.unlinkSync(wrapperScriptPath);
                    console.log('Cleaned up temporary wrapper script');
                }
            } catch (cleanupError) {
                console.warn('Failed to cleanup wrapper script:', cleanupError);
            }
            
            if (code === 0) {
                try {
                    if (!output.trim()) {
                        console.error('Python script produced no output');
                        reject(new Error('Python script produced no output - check if your Python script has the main() wrapper'));
                        return;
                    }
                    
                    // Extract JSON from output that might contain other text
                    const trimmedOutput = output.trim();
                    
                    // Look for JSON in the output - it should start with { or [
                    let jsonStart = -1;
                    let jsonEnd = -1;
                    
                    // Find the first occurrence of { or [
                    for (let i = 0; i < trimmedOutput.length; i++) {
                        if (trimmedOutput[i] === '{' || trimmedOutput[i] === '[') {
                            jsonStart = i;
                            break;
                        }
                    }
                    
                    if (jsonStart === -1) {
                        console.error('No JSON found in Python output:', trimmedOutput);
                        reject(new Error(`Python script produced no JSON output: ${trimmedOutput.substring(0, 200)}...`));
                        return;
                    }
                    
                    // Find the matching closing bracket
                    let bracketCount = 0;
                    let inString = false;
                    let escapeNext = false;
                    const startChar = trimmedOutput[jsonStart];
                    const endChar = startChar === '{' ? '}' : ']';
                    
                    for (let i = jsonStart; i < trimmedOutput.length; i++) {
                        const char = trimmedOutput[i];
                        
                        if (escapeNext) {
                            escapeNext = false;
                            continue;
                        }
                        
                        if (char === '\\') {
                            escapeNext = true;
                            continue;
                        }
                        
                        if (char === '"' && !escapeNext) {
                            inString = !inString;
                            continue;
                        }
                        
                        if (!inString) {
                            if (char === startChar) {
                                bracketCount++;
                            } else if (char === endChar) {
                                bracketCount--;
                                if (bracketCount === 0) {
                                    jsonEnd = i;
                                    break;
                                }
                            }
                        }
                    }
                    
                    if (jsonEnd === -1) {
                        console.error('Incomplete JSON in Python output:', trimmedOutput);
                        reject(new Error(`Python script produced incomplete JSON: ${trimmedOutput.substring(0, 200)}...`));
                        return;
                    }
                    
                    // Extract just the JSON part
                    const jsonString = trimmedOutput.substring(jsonStart, jsonEnd + 1);
                    console.log('Extracted JSON:', jsonString.substring(0, 200) + '...');
                    
                    const results = JSON.parse(jsonString);
                    console.log('Parsed results:', results);

                    // Highlight the smells in the active editor
                    highlightSmells(results, method, filePath);

                    resolve({
                        fileName: path.basename(filePath),
                        method: method,
                        smells: results,
                        filePath: filePath
                    });
                } catch (parseError) {
                    console.error('Failed to parse Python output as JSON:', parseError);
                    console.error('Raw output that failed to parse:', JSON.stringify(output));
                    
                    // Provide more helpful error message
                    const trimmedOutput = output.trim();
                    if (trimmedOutput.includes('not found')) {
                        reject(new Error(`Python script could not find required file: ${trimmedOutput}`));
                    } else if (trimmedOutput.includes('ModuleNotFoundError')) {
                        reject(new Error(`Python module import error: ${trimmedOutput}`));
                    } else {
                        reject(new Error(`Failed to parse analysis results: ${parseError.message}. Raw output: ${trimmedOutput.substring(0, 200)}...`));
                    }
                }
            } else {
                console.error('Python script failed with error:', errorOutput);
                // Combine stdout and stderr for better error reporting
                const fullError = errorOutput || output || 'Unknown error';
                reject(new Error(`Python analysis failed (exit code ${code}): ${fullError}`));
            }
        });
        
        pythonProcess.on('error', (error) => {
            console.error('Failed to start Python process:', error);
            console.error(`Tried to use command: ${pythonCmd}`);
            
            // Clean up the temporary wrapper script
            try {
                if (fs.existsSync(wrapperScriptPath)) {
                    fs.unlinkSync(wrapperScriptPath);
                }
            } catch (cleanupError) {
                console.warn('Failed to cleanup wrapper script:', cleanupError);
            }
            
            if (error.code === 'ENOENT') {
                console.error('Python command not found. Please:');
                console.error('1. Install Python from https://python.org');
                console.error('2. Make sure "Add Python to PATH" is checked during installation');
                console.error('3. Restart VS Code after installing Python');
                console.error('4. Try running "py --version" in Command Prompt to verify installation');
            }
            
            reject(new Error(`Failed to start Python analysis: ${error.message}`));
        });
        
        // Set a timeout to prevent hanging
        setTimeout(() => {
            console.log('Analysis timeout reached, killing process');
            pythonProcess.kill();
            reject(new Error('Analysis timed out after 60 seconds'));
        }, 60000);

    });
}

function getEmbeddedHTML() {
    // This is a fallback HTML content if gui.html doesn't exist
    // You should create the gui.html file in the media folder with the artifact content
    return `
    <!DOCTYPE html>
    <html>
    <head>
        <title>QSpire</title>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
    </head>
    <body>
        <h1>QSpire</h1>
        <p>Please create gui.html in the media folder with your GUI content.</p>
        <script>
            const vscode = acquireVsCodeApi();
        </script>
    </body>
    </html>`;
}

// Listen for active editor changes to update file info
function setupActiveEditorListener() {
    vscode.window.onDidChangeActiveTextEditor(() => {
        if (panel) {
            sendActiveFileInfo();
        }
    });
}

exports.activate = activate;








// Store decoration types to manage them properly
// Store decoration types to manage them properly
let smellDecorationTypes = new Map();

/*
function highlightSmells(results, method, filePath = null) {
    console.log("method method: ",method)

    // Clear any existing decorations first
    clearHighlights();
    
    let editor = vscode.window.activeTextEditor;
    
    // If no active editor, try to open the file that was analyzed
    if (!editor && filePath) {
        // Open the file first
        vscode.workspace.openTextDocument(filePath).then(document => {
            vscode.window.showTextDocument(document).then(newEditor => {
                // Retry highlighting with the newly opened editor
                highlightSmellsWithEditor(results, method, newEditor);
            });
        }).catch(error => {
            vscode.window.showErrorMessage(`Could not open file: ${error.message}`);
        });
        return;
    }
    
    // If still no editor, show error
    if (!editor) {
        vscode.window.showErrorMessage('No active editor found and no file path provided');
        return;
    }
    
    // If we have an editor but filePath is provided, check if it matches
    if (filePath && editor.document.fileName !== filePath) {
        // The active editor is not the file we analyzed, open the correct file
        vscode.workspace.openTextDocument(filePath).then(document => {
            vscode.window.showTextDocument(document).then(newEditor => {
                highlightSmellsWithEditor(results, method, newEditor);
            });
        }).catch(error => {
            // If we can't open the target file, use the current editor anyway
            console.log(`Could not open target file, using active editor: ${error.message}`);
            highlightSmellsWithEditor(results, method, editor);
        });
        return;
    }
    
    // Use the current editor
    highlightSmellsWithEditor(results, method, editor);
}

function highlightSmellsWithEditor(results, method, editor) {

    
    // Access the smells array
    const smells = results.smells;
    
    // Group decorations by type for better performance
    const decorationsByType = new Map();
    
    console.log("Print di debug")
    // Loop through each smell
    smells.forEach((smell, index) => {
        console.log(smell)
        if (method === 'dynamic') {
            highlightDynamicSmell(smell, index, decorationsByType);
        } else if (method === 'static') {
            highlightStaticSmell(smell, index, decorationsByType);
        }
    });

    console.log("Decorations ",decorationsByType)
    
    // Apply all decorations to the editor
    decorationsByType.forEach((ranges, decorationType) => {
        editor.setDecorations(decorationType, ranges);
    });
}
    */


function highlightSmells(results, method, filePath = null) {
    console.log("method method: ", method);
    console.log("filePath to highlight:", filePath);

    // Clear any existing decorations first
    clearHighlights();
    
    // If we have a filePath, find the editor that has this file open
    let editor = null;
    
    if (filePath) {
        // First, check if any visible editor has this file open
        editor = vscode.window.visibleTextEditors.find(e => 
            e.document.uri.fsPath === filePath
        );
        
        console.log("Found visible editor with matching file:", !!editor);
        
        if (editor) {
            // File is already open in a visible editor, use it
            console.log("Using existing visible editor");
            highlightSmellsWithEditor(results, method, editor);
            return;
        }
        
        // If not found in visible editors, try to open/show it
        console.log("File not in visible editors, opening document...");
        vscode.workspace.openTextDocument(filePath).then(document => {
            vscode.window.showTextDocument(document, {
                viewColumn: vscode.ViewColumn.One, // Open in the first column
                preserveFocus: false, // Give it focus
                preview: false // Don't open as preview (which might replace other tabs)
            }).then(newEditor => {
                console.log("Document opened, applying highlights");
                highlightSmellsWithEditor(results, method, newEditor);
            });
        }).catch(error => {
            vscode.window.showErrorMessage(`Could not open file: ${error.message}`);
        });
        return;
    }
    
    // Fallback: use active editor if no filePath provided
    editor = vscode.window.activeTextEditor;
    
    if (!editor) {
        vscode.window.showErrorMessage('No active editor found and no file path provided');
        return;
    }
    
    highlightSmellsWithEditor(results, method, editor);
}


function highlightSmellsWithEditor(results, method, editor) {
    // Access the smells array
    const smells = results.smells;
    
    // Group decorations by type for better performance
    const decorationsByType = new Map();
    
    console.log("Print di debug");
    // Loop through each smell
    smells.forEach((smell, index) => {
        console.log(smell);
        if (method === 'dynamic') {
            highlightDynamicSmell(smell, index, decorationsByType, editor);
        } else if (method === 'static') {
            highlightStaticSmell(smell, index, decorationsByType, editor);
        }
    });

    console.log("Decorations ", decorationsByType);
    
    // Apply all decorations to the editor
    decorationsByType.forEach((ranges, decorationType) => {
        editor.setDecorations(decorationType, ranges);
    });
    
    // Optional: Make sure the editor is visible and focused
    vscode.window.showTextDocument(editor.document, {
        viewColumn: editor.viewColumn,
        preserveFocus: false
    });
}


/*
function highlightSmellsWithEditor(results, method, editor) {
    // Access the smells array
    const smells = results.smells;
    
    // Group decorations by type for better performance
    const decorationsByType = new Map();
    
    console.log("Print di debug");
    // Loop through each smell - PASS EDITOR to the highlight functions
    smells.forEach((smell, index) => {
        console.log(smell);
        if (method === 'dynamic') {
            highlightDynamicSmell(smell, index, decorationsByType, editor);
        } else if (method === 'static') {
            highlightStaticSmell(smell, index, decorationsByType, editor);
        }
    });

    console.log("Decorations ", decorationsByType);
    
    // Apply all decorations to the editor
    decorationsByType.forEach((ranges, decorationType) => {
        editor.setDecorations(decorationType, ranges);
    });
    
    // Optional: Make sure the editor is visible and focused
    vscode.window.showTextDocument(editor.document, {
        viewColumn: editor.viewColumn,
        preserveFocus: false
    });
}
*/


function highlightDynamicSmell(smell, smellIndex, decorationsByType, editor) {
    
    switch (smell.type) {


        case 'CG':
            // Handle CG smell: highlight from column_start to column_end on the specified row
            if (smell.row !== undefined && smell.column_start !== undefined && smell.column_end !== undefined) {
                const decorationType = getCGDecorationType();
                
                if (!decorationsByType.has(decorationType)) {
                    decorationsByType.set(decorationType, []);
                }
                
                const range = new vscode.Range(
                    smell.row - 1, // Convert to 0-based indexing
                    smell.column_start - 1,
                    smell.row - 1,
                    smell.column_end - 1
                );
                
                decorationsByType.get(decorationType).push({
                    range: range,
                    hoverMessage: new vscode.MarkdownString(`**Code Smell: ${smell.type}**\n\n` +
                        `Row: ${smell.row}\n\n` +
                        `Columns: ${smell.column_start}-${smell.column_end}\n\n` +
                        `${smell.description || 'CG smell detected'}`)
                });
            }
            break;

        
        case 'LPQ':
            // Handle CG smell: highlight from column_start to column_end on the specified row
            if (smell.row !== undefined && smell.column_start !== undefined && smell.column_end !== undefined) {
                const decorationType = getLPQDecorationType();
                
                if (!decorationsByType.has(decorationType)) {
                    decorationsByType.set(decorationType, []);
                }
                
                const range = new vscode.Range(
                    smell.row - 1, // Convert to 0-based indexing
                    smell.column_start - 1,
                    smell.row - 1,
                    smell.column_end - 1
                );
                
                decorationsByType.get(decorationType).push({
                    range: range,
                    hoverMessage: new vscode.MarkdownString(`**Code Smell: ${smell.type}**\n\n` +
                        `Row: ${smell.row}\n\n` +
                        `Columns: ${smell.column_start}-${smell.column_end}\n\n` +
                        `${smell.description || 'LPQ smell detected'}`)
                });
            }
            break;


        case 'IdQ':
            // Handle IQ smell: highlight from column_start to column_end on the specified row
            if (smell.row !== undefined && smell.column_start !== undefined && smell.column_end !== undefined) {
                // Debug: Log the actual values
                console.log('Smell data:', {
                    type: smell.type,
                    row: smell.row,
                    column_start: smell.column_start,
                    column_end: smell.column_end
                });

                // Validate that all values are positive after conversion
                const row = Math.max(0, (smell.row || 1) - 1);
                const colStart = Math.max(0, (smell.column_start || 1) - 1);
                const colEnd = Math.max(0, (smell.column_end || 1) - 1);
                
                console.log('Converted values:', { row, colStart, colEnd });

                const decorationType = getIdQDecorationType();
                
                if (!decorationsByType.has(decorationType)) {
                    decorationsByType.set(decorationType, []);
                }
                
                const range = new vscode.Range(row, colStart, row, colEnd);
                
                decorationsByType.get(decorationType).push({
                    range: range,
                    hoverMessage: new vscode.MarkdownString(`**Code Smell: ${smell.type}**\n\n` +
                        `Row: ${smell.row}\n\n` +
                        `Columns: ${smell.column_start}-${smell.column_end}\n\n` +
                        `${smell.description || 'IdQ smell detected'}`)
                });
            } else {
                console.log('Missing required fields:', smell);
            }
            break;

        /*
        case 'IdQ':
            // Handle CG smell: highlight from column_start to column_end on the specified row
            if (smell.row !== undefined && smell.column_start !== undefined && smell.column_end !== undefined) {
                const decorationType = getIdQDecorationType();
                
                if (!decorationsByType.has(decorationType)) {
                    decorationsByType.set(decorationType, []);
                }
                
                const range = new vscode.Range(
                    smell.row - 1, // Convert to 0-based indexing
                    smell.column_start - 1,
                    smell.row - 1,
                    smell.column_end - 1
                );
                
                decorationsByType.get(decorationType).push({
                    range: range,
                    hoverMessage: new vscode.MarkdownString(`**Code Smell: ${smell.type}**\n\n` +
                        `Row: ${smell.row}\n\n` +
                        `Columns: ${smell.column_start}-${smell.column_end}\n\n` +
                        `${smell.description || 'IdQ smell detected'}`)
                });
            }
            break;
        */

        /*
        case 'IQ':
            // Handle CG smell: highlight from column_start to column_end on the specified row
            if (smell.row !== undefined && smell.column_start !== undefined && smell.column_end !== undefined) {
                const decorationType = getIQDecorationType();
                
                if (!decorationsByType.has(decorationType)) {
                    decorationsByType.set(decorationType, []);
                }
                
                const range = new vscode.Range(
                    smell.row - 1, // Convert to 0-based indexing
                    smell.column_start - 1,
                    smell.row - 1,
                    smell.column_end - 1
                );
                
                decorationsByType.get(decorationType).push({
                    range: range,
                    hoverMessage: new vscode.MarkdownString(`**Code Smell: ${smell.type}**\n\n` +
                        `Row: ${smell.row}\n\n` +
                        `Columns: ${smell.column_start}-${smell.column_end}\n\n` +
                        `${smell.description || 'IQ smell detected'}`)
                });
            }
            break;
        */

        case 'IQ':
            // Handle IQ smell: highlight from column_start to column_end on the specified row
            if (smell.row !== undefined && smell.column_start !== undefined && smell.column_end !== undefined) {
                // Debug: Log the actual values
                console.log('Smell data:', {
                    type: smell.type,
                    row: smell.row,
                    column_start: smell.column_start,
                    column_end: smell.column_end
                });

                // Validate that all values are positive after conversion
                const row = Math.max(0, (smell.row || 1) - 1);
                const colStart = Math.max(0, (smell.column_start || 1) - 1);
                const colEnd = Math.max(0, (smell.column_end || 1) - 1);
                
                console.log('Converted values:', { row, colStart, colEnd });

                const decorationType = getIQDecorationType();
                
                if (!decorationsByType.has(decorationType)) {
                    decorationsByType.set(decorationType, []);
                }
                
                const range = new vscode.Range(row, colStart, row, colEnd);
                
                decorationsByType.get(decorationType).push({
                    range: range,
                    hoverMessage: new vscode.MarkdownString(`**Code Smell: ${smell.type}**\n\n` +
                        `Row: ${smell.row}\n\n` +
                        `Columns: ${smell.column_start}-${smell.column_end}\n\n` +
                        `${smell.description || 'IQ smell detected'}`)
                });
            } else {
                console.log('Missing required fields:', smell);
            }
            break;

        
        case 'IM':
            // Handle CG smell: highlight from column_start to column_end on the specified row
            if (smell.row !== undefined && smell.column_start !== undefined && smell.column_end !== undefined) {
                const decorationType = getIMDecorationType();
                
                if (!decorationsByType.has(decorationType)) {
                    decorationsByType.set(decorationType, []);
                }
                
                const range = new vscode.Range(
                    smell.row - 1, // Convert to 0-based indexing
                    smell.column_start - 1,
                    smell.row - 1,
                    smell.column_end - 1
                );
                
                decorationsByType.get(decorationType).push({
                    range: range,
                    hoverMessage: new vscode.MarkdownString(`**Code Smell: ${smell.type}**\n\n` +
                        `Row: ${smell.row}\n\n` +
                        `Columns: ${smell.column_start}-${smell.column_end}\n\n` +
                        `${smell.description || 'IM smell detected'}`)
                });
            }
            break;


        case 'ROC':
            console.log('ROC case triggered');
            console.log('smell.rows:', smell.rows);
            
            // Handle ROC smell: highlight multiple rows on the side without text highlighting
            if (smell.rows !== undefined && Array.isArray(smell.rows)) {
                console.log('ROC rows is array');
                const decorationType = getROCDecorationType();
                
                if (!decorationsByType.has(decorationType)) {
                    decorationsByType.set(decorationType, []);
                }
                
                // Extract all unique row numbers from various tuple formats
                const allRowNumbers = [];
                
                /*
                smell.rows.forEach(rowStr => {
                    // Remove parentheses and split by comma to get all numbers
                    const cleanStr = rowStr.replace(/[()]/g, ''); // Remove ( and )
                    const numbers = cleanStr.split(',')
                        .map(num => num.trim()) // Remove whitespace
                        .filter(num => num !== '') // Remove empty strings
                        .map(num => parseInt(num, 10)) // Convert to integers
                        .filter(num => !isNaN(num)); // Remove any NaN values
                    
                    allRowNumbers.push(...numbers);
                });*/

                smell.rows.forEach(item => {
                    if (typeof item === "number") {
                        allRowNumbers.push(item);
                    } else {
                        // treat it as a string representation "(9,10)"
                        const numbers = item
                            .replace(/[()]/g, "")
                            .split(",")
                            .map(n => parseInt(n.trim(), 10))
                            .filter(n => !isNaN(n));
                        allRowNumbers.push(...numbers);
                    }
                });
                                
                // Get unique row numbers
                const uniqueRows = [...new Set(allRowNumbers)];
                
                console.log('All row numbers:', allRowNumbers);
                console.log('Unique rows:', uniqueRows);
                
                // Create decorations for each unique row
                uniqueRows.forEach(rowNumber => {
                    console.log('Creating decoration for row:', rowNumber);
                    
                    // Get the full line range to ensure the decoration is visible
                    const startPos = new vscode.Position(rowNumber - 1, 0);
                    const endPos = new vscode.Position(rowNumber - 1, Number.MAX_SAFE_INTEGER);
                    const range = new vscode.Range(startPos, endPos);
                    
                    console.log('Range created:', range);
                    
                    decorationsByType.get(decorationType).push({
                        range: range,
                        hoverMessage: new vscode.MarkdownString(`**Code Smell: ${smell.type}**\n\n` +
                            `Affected rows: ${uniqueRows.join(', ')}\n\n` +
                            `${smell.description || 'ROC smell detected'}`)
                    });
                    console.log("Change ROC made");
                });
                
                console.log('Total ROC decorations:', decorationsByType.get(decorationType).length);
            } else {
                console.log('ROC rows is not an array or undefined');
            }
            break;


        /*
        case 'LC':
            // Handle LC smell: show warning at the end of the file
            console.log('LC case triggered');
            const decorationType = getLCDecorationType();
            
            if (!decorationsByType.has(decorationType)) {
                decorationsByType.set(decorationType, []);
            }
            
            // Get the last line of the document
            if (vscode.window.activeTextEditor) {
                const document = vscode.window.activeTextEditor.document;
                const lastLineNumber = document.lineCount - 1;
                
                // Create a range that covers the entire last line
                const range = new vscode.Range(
                    lastLineNumber,
                    0,
                    lastLineNumber,
                    Number.MAX_SAFE_INTEGER
                );
                
                decorationsByType.get(decorationType).push({
                    range: range,
                    hoverMessage: new vscode.MarkdownString(`**Code Smell: ${smell.type}**\n\n` +
                        `${smell.description || 'LC smell detected'}\n\n` +
                        `This smell affects the entire file.`)
                });
                
                console.log('LC decoration created on last line:', lastLineNumber + 1);
            }
            break;
        */

        
        case 'LC':
        // Handle LC smell: show warning at the end of the file
        console.log('LC case triggered');
        const decorationType = getLCDecorationType();
        
        if (!decorationsByType.has(decorationType)) {
            decorationsByType.set(decorationType, []);
        }
        
        // Use the passed editor parameter instead of activeTextEditor
        if (editor && editor.document) {
            const document = editor.document;
            const lastLineNumber = document.lineCount - 1;
            
            // Create a range that covers the entire last line
            const range = new vscode.Range(
                lastLineNumber,
                0,
                lastLineNumber,
                Number.MAX_SAFE_INTEGER
            );
            
            decorationsByType.get(decorationType).push({
                range: range,
                hoverMessage: new vscode.MarkdownString(`**Code Smell: ${smell.type}**\n\n` +
                    `${smell.description || 'LC smell detected'}\n\n` +
                    `This smell affects the whole circuit.`)
            });
            
            console.log('LC decoration created on last line:', lastLineNumber + 1);
        } else {
            console.error('No editor available for LC decoration');
        }
        break;



        

        case 'NC':
            console.log('NC case triggered');
            console.log('NC smell object:', smell);
            
            // Handle NC smell: extract rows from all call types
            const NCdecorationType = getNCDecorationType();
            
            if (!decorationsByType.has(NCdecorationType)) {
                decorationsByType.set(NCdecorationType, []);
            }
            
            // Extract all row numbers from various call arrays
            const allRowNumbers = [];
            
            // Extract from run_calls array
            if (smell.run_calls && Array.isArray(smell.run_calls)) {
                smell.run_calls.forEach(call => {
                    if (call.row && typeof call.row === 'number') {
                        allRowNumbers.push(call.row);
                    }
                });
            }
            
            // Extract from execute_calls (could be array or object)
            if (smell.execute_calls) {
                if (Array.isArray(smell.execute_calls)) {
                    smell.execute_calls.forEach(call => {
                        if (call.row && typeof call.row === 'number') {
                            allRowNumbers.push(call.row);
                        }
                    });
                } else if (typeof smell.execute_calls === 'object') {
                    // If it's an object, look for arrays inside it
                    Object.values(smell.execute_calls).forEach(calls => {
                        if (Array.isArray(calls)) {
                            calls.forEach(call => {
                                if (call.row && typeof call.row === 'number') {
                                    allRowNumbers.push(call.row);
                                }
                            });
                        }
                    });
                }
            }
            
            // Extract from assign_parameter_calls array
            if (smell.assign_parameter_calls && Array.isArray(smell.assign_parameter_calls)) {
                smell.assign_parameter_calls.forEach(call => {
                    if (call.row && typeof call.row === 'number') {
                        allRowNumbers.push(call.row);
                    }
                });
            }
            
            // Extract from bind_parameter_calls (could be array or object)
            if (smell.bind_parameter_calls) {
                if (Array.isArray(smell.bind_parameter_calls)) {
                    smell.bind_parameter_calls.forEach(call => {
                        if (call.row && typeof call.row === 'number') {
                            allRowNumbers.push(call.row);
                        }
                    });
                } else if (typeof smell.bind_parameter_calls === 'object') {
                    // If it's an object, look for arrays inside it
                    Object.values(smell.bind_parameter_calls).forEach(calls => {
                        if (Array.isArray(calls)) {
                            calls.forEach(call => {
                                if (call.row && typeof call.row === 'number') {
                                    allRowNumbers.push(call.row);
                                }
                            });
                        }
                    });
                }
            }
            
            // Get unique row numbers
            const uniqueRows = [...new Set(allRowNumbers)];
            
            console.log('All row numbers from NC calls:', allRowNumbers);
            console.log('Unique NC rows:', uniqueRows);
            
            // Create decorations for each unique row
            uniqueRows.forEach(rowNumber => {
                console.log('Creating NC decoration for row:', rowNumber);
                
                // Get the full line range
                const startPos = new vscode.Position(rowNumber - 1, 0);
                const endPos = new vscode.Position(rowNumber - 1, Number.MAX_SAFE_INTEGER);
                const range = new vscode.Range(startPos, endPos);
                
                decorationsByType.get(NCdecorationType).push({
                    range: range,
                    hoverMessage: new vscode.MarkdownString(`**Code Smell: ${smell.type}**\n\n` +
                        `Affected rows: ${uniqueRows.join(', ')}\n\n` +
                        `Run count: ${smell.run_count || 0}\n\n` +
                        `Execute count: ${smell.execute_count || 0}\n\n` +
                        `Assign parameters count: ${smell.assign_parameters_count || 0}\n\n` +
                        `Bind parameters count: ${smell.bind_parameters_count || 0}\n\n` +
                        `${smell.explanation || 'NC smell detected'}`)
                });
            });
            
            console.log('Total NC decorations:', decorationsByType.get(NCdecorationType).length);
            break;
    
    
        default:
            console.log(`Unknown dynamic smell type: ${smell.type}`);
    }
}



function getCGDecorationType() {
    if (!smellDecorationTypes.has('CG')) {
        const decorationType = vscode.window.createTextEditorDecorationType({
            border: '2px solid orange',
            borderRadius: '3px',
            overviewRulerColor: 'orange',
            overviewRulerLane: vscode.OverviewRulerLane.Right,
            after: {
                contentText: ' ⚠️ CG',
                color: 'orange',
                fontWeight: 'bold',
                margin: '0 0 0 10px'
            }
        });
        smellDecorationTypes.set('CG', decorationType);
    }
    return smellDecorationTypes.get('CG');
}

function getLPQDecorationType() {
    if (!smellDecorationTypes.has('LPQ')) {
        const decorationType = vscode.window.createTextEditorDecorationType({
            border: '2px solid orange',
            borderRadius: '3px',
            overviewRulerColor: 'orange',
            overviewRulerLane: vscode.OverviewRulerLane.Right,
            after: {
                contentText: ' ⚠️ LPQ',
                color: 'orange',
                fontWeight: 'bold',
                margin: '0 0 0 10px'
            }
        });
        smellDecorationTypes.set('LPQ', decorationType);
    }
    return smellDecorationTypes.get('LPQ');
}

function getIdQDecorationType() {
    if (!smellDecorationTypes.has('IdQ')) {
        const decorationType = vscode.window.createTextEditorDecorationType({
            border: '2px solid orange',
            borderRadius: '3px',
            overviewRulerColor: 'orange',
            overviewRulerLane: vscode.OverviewRulerLane.Right,
            after: {
                contentText: ' ⚠️ IdQ',
                color: 'orange',
                fontWeight: 'bold',
                margin: '0 0 0 10px'
            }
        });
        smellDecorationTypes.set('IdQ', decorationType);
    }
    return smellDecorationTypes.get('IdQ');
}

function getIQDecorationType() {
    if (!smellDecorationTypes.has('IQ')) {
        const decorationType = vscode.window.createTextEditorDecorationType({
            border: '2px solid orange',
            borderRadius: '3px',
            overviewRulerColor: 'orange',
            overviewRulerLane: vscode.OverviewRulerLane.Right,
            after: {
                contentText: ' ⚠️ IQ',
                color: 'orange',
                fontWeight: 'bold',
                margin: '0 0 0 10px'
            }
        });
        smellDecorationTypes.set('IQ', decorationType);
    }
    return smellDecorationTypes.get('IQ');
}

function getIMDecorationType() {
    if (!smellDecorationTypes.has('IM')) {
        const decorationType = vscode.window.createTextEditorDecorationType({
            border: '2px solid orange',
            borderRadius: '3px',
            overviewRulerColor: 'orange',
            overviewRulerLane: vscode.OverviewRulerLane.Right,
            after: {
                contentText: ' ⚠️ IM',
                color: 'orange',
                fontWeight: 'bold',
                margin: '0 0 0 10px'
            }
        });
        smellDecorationTypes.set('IM', decorationType);
    }
    return smellDecorationTypes.get('IM');
}


function getROCDecorationType() {
    if (!smellDecorationTypes.has('ROC')) {
        const decorationType = vscode.window.createTextEditorDecorationType({
             backgroundColor: 'rgba(255, 255, 0, 0.15)', // Light yellow with 15% opacity
            // No background or border highlighting
            // Just add a warning at the end of the line
            after: {
                contentText: ' ⚠️ ROC',
                color: 'orange',
                fontWeight: 'bold',
                margin: '0 0 0 10px'
            },
            // Optional: Keep the overview ruler for navigation
            overviewRulerColor: 'yellow',
            overviewRulerLane: vscode.OverviewRulerLane.Right
        });
        smellDecorationTypes.set('ROC', decorationType);
    }
    return smellDecorationTypes.get('ROC');
}

function getLCDecorationType() {
    if (!smellDecorationTypes.has('LC')) {
        const decorationType = vscode.window.createTextEditorDecorationType({
            // Add warning after the last line content
            after: {
                contentText: ' ⚠️ LC',
                color: 'red',
                fontWeight: 'bold',
                margin: '0 0 0 20px'
            },
            // Add subtle background to the entire last line
            backgroundColor: 'rgba(255, 0, 0, 0.1)',
            // Add to overview ruler
            overviewRulerColor: 'yellow',
            overviewRulerLane: vscode.OverviewRulerLane.Right
        });
        smellDecorationTypes.set('LC', decorationType);
    }
    return smellDecorationTypes.get('LC');
}

function getNCDecorationType() {
    if (!smellDecorationTypes.has('NC')) {
        const decorationType = vscode.window.createTextEditorDecorationType({
             backgroundColor: 'rgba(255, 255, 0, 0.15)', // Light yellow with 15% opacity
            // Add warning at the end of affected lines
            after: {
                contentText: ' 🔄 NC',
                color: 'purple',
                fontWeight: 'bold',
                margin: '0 0 0 10px'
            },
            // Add to overview ruler
            overviewRulerColor: 'purple',
            overviewRulerLane: vscode.OverviewRulerLane.Right
        });
        smellDecorationTypes.set('NC', decorationType);
    }
    return smellDecorationTypes.get('NC');
}
















function highlightStaticSmell(smell, smellIndex, decorationsByType, editor) {
    // Different highlighting logic for static method
    switch (smell.type) {

        case 'CG':
            // Handle CG smell: highlight from column_start to column_end on the specified row
            if (smell.row !== undefined && smell.column_start !== undefined && smell.column_end !== undefined) {
                const decorationType = getCGDecorationType();
                
                if (!decorationsByType.has(decorationType)) {
                    decorationsByType.set(decorationType, []);
                }
                
                const range = new vscode.Range(
                    smell.row - 1, // Convert to 0-based indexing
                    smell.column_start - 1,
                    smell.row - 1,
                    smell.column_end - 1
                );
                
                decorationsByType.get(decorationType).push({
                    range: range,
                    hoverMessage: new vscode.MarkdownString(`**Code Smell: ${smell.type}**\n\n` +
                        `Row: ${smell.row}\n\n` +
                        `Columns: ${smell.column_start}-${smell.column_end}\n\n` +
                        `${smell.description || 'CG smell detected'}`)
                });
            }
            break;

        
        case 'LPQ':
            // Handle CG smell: highlight from column_start to column_end on the specified row
            if (smell.row !== undefined && smell.column_start !== undefined && smell.column_end !== undefined) {
                const decorationType = getLPQDecorationType();
                
                if (!decorationsByType.has(decorationType)) {
                    decorationsByType.set(decorationType, []);
                }
                
                const range = new vscode.Range(
                    smell.row - 1, // Convert to 0-based indexing
                    smell.column_start - 1,
                    smell.row - 1,
                    smell.column_end - 1
                );
                
                decorationsByType.get(decorationType).push({
                    range: range,
                    hoverMessage: new vscode.MarkdownString(`**Code Smell: ${smell.type}**\n\n` +
                        `Row: ${smell.row}\n\n` +
                        `Columns: ${smell.column_start}-${smell.column_end}\n\n` +
                        `${smell.description || 'LPQ smell detected'}`)
                });
            }
            break;



        case 'IdQ':
            // Handle IdQ smell: highlight only the row, not specific columns
            if (smell.row !== undefined) {
                const decorationType = getStaticIdQDecorationType();
                
                if (!decorationsByType.has(decorationType)) {
                    decorationsByType.set(decorationType, []);
                }

                try{
                
                    const range = new vscode.Range(
                        smell.row - 1, // Convert to 0-based indexing
                        0, // Start at beginning of line
                        smell.row - 1, // Same row
                        Number.MAX_SAFE_INTEGER // End at end of line
                    );
                    
                    decorationsByType.get(decorationType).push({
                        range: range,
                        hoverMessage: new vscode.MarkdownString(`**Code Smell: ${smell.type}**\n\n` +
                            `Row: ${smell.row}\n\n` +
                            `${smell.description || 'IdQ smell detected'}`)
                    });
                }catch(err){;}

            }
            break;



        case 'IQ':
            // Handle IdQ smell: highlight only the row, not specific columns
            if (smell.row !== undefined) {
                const decorationType = getStaticIQDecorationType();
                
                if (!decorationsByType.has(decorationType)) {
                    decorationsByType.set(decorationType, []);
                }
                
                try{
                    const range = new vscode.Range(
                        smell.row - 1, // Convert to 0-based indexing
                        0, // Start at beginning of line
                        smell.row - 1, // Same row
                        Number.MAX_SAFE_INTEGER // End at end of line
                    );
                    
                    decorationsByType.get(decorationType).push({
                        range: range,
                        hoverMessage: new vscode.MarkdownString(`**Code Smell: ${smell.type}**\n\n` +
                            `Row: ${smell.row}\n\n` +
                            `${smell.description || 'IQ smell detected'}`)
                    });
                }catch(err){;}
            }
            break;

        
        case 'IM':
            // Handle IdQ smell: highlight only the row, not specific columns
            if (smell.row !== undefined) {
                const decorationType = getStaticIMDecorationType();
                
                if (!decorationsByType.has(decorationType)) {
                    decorationsByType.set(decorationType, []);
                }
                
                try{

                    const range = new vscode.Range(
                        smell.row - 1, // Convert to 0-based indexing
                        0, // Start at beginning of line
                        smell.row - 1, // Same row
                        Number.MAX_SAFE_INTEGER // End at end of line
                    );
                    
                    decorationsByType.get(decorationType).push({
                        range: range,
                        hoverMessage: new vscode.MarkdownString(`**Code Smell: ${smell.type}**\n\n` +
                            `Row: ${smell.row}\n\n` +
                            `${smell.description || 'IM smell detected'}`)
                    });
                }catch(err){;}
            }
            break;


        case 'ROC':
            console.log('ROC case triggered');
            console.log('smell.rows:', smell.rows);
            
            // Handle ROC smell: highlight multiple rows on the side without text highlighting
            if (smell.rows !== undefined && Array.isArray(smell.rows)) {
                console.log('ROC rows is array');
                const decorationType = getROCDecorationType();
                
                if (!decorationsByType.has(decorationType)) {
                    decorationsByType.set(decorationType, []);
                }
                
                // Extract all unique row numbers from various tuple formats
                const allRowNumbers = [];
                
                /*
                smell.rows.forEach(rowStr => {
                    // Remove parentheses and split by comma to get all numbers
                    const cleanStr = rowStr.replace(/[()]/g, ''); // Remove ( and )
                    const numbers = cleanStr.split(',')
                        .map(num => num.trim()) // Remove whitespace
                        .filter(num => num !== '') // Remove empty strings
                        .map(num => parseInt(num, 10)) // Convert to integers
                        .filter(num => !isNaN(num)); // Remove any NaN values
                    
                    allRowNumbers.push(...numbers);
                });*/

                smell.rows.forEach(item => {
                    if (typeof item === "number") {
                        allRowNumbers.push(item);
                    } else {
                        // treat it as a string representation "(9,10)"
                        const numbers = item
                            .replace(/[()]/g, "")
                            .split(",")
                            .map(n => parseInt(n.trim(), 10))
                            .filter(n => !isNaN(n));
                        allRowNumbers.push(...numbers);
                    }
                });
                                
                // Get unique row numbers
                const uniqueRows = [...new Set(allRowNumbers)];
                
                console.log('All row numbers:', allRowNumbers);
                console.log('Unique rows:', uniqueRows);
                
                // Create decorations for each unique row
                uniqueRows.forEach(rowNumber => {

                    try{
                        console.log('Creating decoration for row:', rowNumber);
                        
                        // Get the full line range to ensure the decoration is visible
                        const startPos = new vscode.Position(rowNumber - 1, 0);
                        const endPos = new vscode.Position(rowNumber - 1, Number.MAX_SAFE_INTEGER);
                        const range = new vscode.Range(startPos, endPos);
                        
                        console.log('Range created:', range);
                        
                        decorationsByType.get(decorationType).push({
                            range: range,
                            hoverMessage: new vscode.MarkdownString(`**Code Smell: ${smell.type}**\n\n` +
                                `Affected rows: ${uniqueRows.join(', ')}\n\n` +
                                `${smell.description || 'ROC smell detected'}`)
                        });
                        console.log("Change ROC made");
                    }catch(err){;}
                });
                
                console.log('Total ROC decorations:', decorationsByType.get(decorationType).length);
            } else {
                console.log('ROC rows is not an array or undefined');
            }
            break;


        /*
        case 'LC':
            // Handle LC smell: show warning at the end of the file
            console.log('LC case triggered');
            const decorationType = getLCDecorationType();
            
            if (!decorationsByType.has(decorationType)) {
                decorationsByType.set(decorationType, []);
            }
            
            // Get the last line of the document
            if (vscode.window.activeTextEditor) {
                const document = vscode.window.activeTextEditor.document;
                const lastLineNumber = document.lineCount - 1;
                
                // Create a range that covers the entire last line
                const range = new vscode.Range(
                    lastLineNumber,
                    0,
                    lastLineNumber,
                    Number.MAX_SAFE_INTEGER
                );
                
                decorationsByType.get(decorationType).push({
                    range: range,
                    hoverMessage: new vscode.MarkdownString(`**Code Smell: ${smell.type}**\n\n` +
                        `${smell.description || 'LC smell detected'}\n\n` +
                        `This smell affects the entire file.`)
                });
                
                console.log('LC decoration created on last line:', lastLineNumber + 1);
            }
            break;
        */


        case 'LC':
            // Handle LC smell: show warning at the end of the file
            console.log('LC case triggered');
            const decorationType = getLCDecorationType();
            
            if (!decorationsByType.has(decorationType)) {
                decorationsByType.set(decorationType, []);
            }
            
            // Use the passed editor parameter instead of activeTextEditor
            if (editor && editor.document) {
                const document = editor.document;
                const lastLineNumber = document.lineCount - 1;
                
                // Create a range that covers the entire last line
                const range = new vscode.Range(
                    lastLineNumber,
                    0,
                    lastLineNumber,
                    Number.MAX_SAFE_INTEGER
                );
                
                decorationsByType.get(decorationType).push({
                    range: range,
                    hoverMessage: new vscode.MarkdownString(`**Code Smell: ${smell.type}**\n\n` +
                        `${smell.description || 'LC smell detected'}\n\n` +
                        `This smell affects the whole circuit.`)
                });
                
                console.log('LC decoration created on last line:', lastLineNumber + 1);
            } else {
                console.error('No editor available for LC decoration');
            }
            break;
        

        case 'NC':
            console.log('NC case triggered');
            console.log('NC smell object:', smell);
            
            // Handle NC smell: extract rows from all call types
            const NCdecorationType = getNCDecorationType();
            
            if (!decorationsByType.has(NCdecorationType)) {
                decorationsByType.set(NCdecorationType, []);
            }
            
            // Extract all row numbers from various call arrays
            const allRowNumbers = [];
            
            // Extract from run_calls array
            if (smell.run_calls && Array.isArray(smell.run_calls)) {
                smell.run_calls.forEach(call => {
                    if (call.row && typeof call.row === 'number') {
                        allRowNumbers.push(call.row);
                    }
                });
            }
            
            // Extract from execute_calls (could be array or object)
            if (smell.execute_calls) {
                if (Array.isArray(smell.execute_calls)) {
                    smell.execute_calls.forEach(call => {
                        if (call.row && typeof call.row === 'number') {
                            allRowNumbers.push(call.row);
                        }
                    });
                } else if (typeof smell.execute_calls === 'object') {
                    // If it's an object, look for arrays inside it
                    Object.values(smell.execute_calls).forEach(calls => {
                        if (Array.isArray(calls)) {
                            calls.forEach(call => {
                                if (call.row && typeof call.row === 'number') {
                                    allRowNumbers.push(call.row);
                                }
                            });
                        }
                    });
                }
            }
            
            // Extract from assign_parameter_calls array
            if (smell.assign_parameter_calls && Array.isArray(smell.assign_parameter_calls)) {
                smell.assign_parameter_calls.forEach(call => {
                    if (call.row && typeof call.row === 'number') {
                        allRowNumbers.push(call.row);
                    }
                });
            }
            
            // Extract from bind_parameter_calls (could be array or object)
            if (smell.bind_parameter_calls) {
                if (Array.isArray(smell.bind_parameter_calls)) {
                    smell.bind_parameter_calls.forEach(call => {
                        if (call.row && typeof call.row === 'number') {
                            allRowNumbers.push(call.row);
                        }
                    });
                } else if (typeof smell.bind_parameter_calls === 'object') {
                    // If it's an object, look for arrays inside it
                    Object.values(smell.bind_parameter_calls).forEach(calls => {
                        if (Array.isArray(calls)) {
                            calls.forEach(call => {
                                if (call.row && typeof call.row === 'number') {
                                    allRowNumbers.push(call.row);
                                }
                            });
                        }
                    });
                }
            }
            
            // Get unique row numbers
            const uniqueRows = [...new Set(allRowNumbers)];
            
            console.log('All row numbers from NC calls:', allRowNumbers);
            console.log('Unique NC rows:', uniqueRows);
            
            // Create decorations for each unique row
            uniqueRows.forEach(rowNumber => {

                try{
                    console.log('Creating NC decoration for row:', rowNumber);
                    
                    // Get the full line range
                    const startPos = new vscode.Position(rowNumber - 1, 0);
                    const endPos = new vscode.Position(rowNumber - 1, Number.MAX_SAFE_INTEGER);
                    const range = new vscode.Range(startPos, endPos);
                    
                    decorationsByType.get(NCdecorationType).push({
                        range: range,
                        hoverMessage: new vscode.MarkdownString(`**Code Smell: ${smell.type}**\n\n` +
                            `Affected rows: ${uniqueRows.join(', ')}\n\n` +
                            `Run count: ${smell.run_count || 0}\n\n` +
                            `Execute count: ${smell.execute_count || 0}\n\n` +
                            `Assign parameters count: ${smell.assign_parameters_count || 0}\n\n` +
                            `Bind parameters count: ${smell.bind_parameters_count || 0}\n\n` +
                            `${smell.explanation || 'NC smell detected'}`)
                    });
                }catch(err){;}
            });
            
            console.log('Total NC decorations:', decorationsByType.get(NCdecorationType).length);
            break;
    
    
            
        default:
            console.log(`Unknown static smell type: ${smell.type}`);
    }
}




function getStaticIdQDecorationType() {
    if (!smellDecorationTypes.has('IdQ')) {
        const decorationType = vscode.window.createTextEditorDecorationType({
             backgroundColor: 'rgba(255, 255, 0, 0.15)', // Light yellow with 15% opacity
            // Add warning at the end of the line
            after: {
                contentText: ' 🎯 IdQ',
                color: 'blue',
                fontWeight: 'bold',
                margin: '0 0 0 10px'
            },
            // Add to overview ruler
            overviewRulerColor: 'blue',
            overviewRulerLane: vscode.OverviewRulerLane.Right
        });
        smellDecorationTypes.set('IdQ', decorationType);
    }
    return smellDecorationTypes.get('IdQ');
}

function getStaticIQDecorationType() {
    if (!smellDecorationTypes.has('IQ')) {
        const decorationType = vscode.window.createTextEditorDecorationType({
             backgroundColor: 'rgba(255, 255, 0, 0.15)', // Light yellow with 15% opacity
            // Add warning at the end of the line
            after: {
                contentText: ' 🎯 IQ',
                color: 'blue',
                fontWeight: 'bold',
                margin: '0 0 0 10px'
            },
            // Add to overview ruler
            overviewRulerColor: 'blue',
            overviewRulerLane: vscode.OverviewRulerLane.Right
        });
        smellDecorationTypes.set('IQ', decorationType);
    }
    return smellDecorationTypes.get('IQ');
}

function getStaticIMDecorationType() {
    if (!smellDecorationTypes.has('IM')) {
        const decorationType = vscode.window.createTextEditorDecorationType({
            // Add gentle yellow background to the whole line
            backgroundColor: 'rgba(255, 255, 0, 0.15)', // Light yellow with 15% opacity
            // Add warning at the end of the line
            after: {
                contentText: ' 🎯 IM',
                color: 'blue',
                fontWeight: 'bold',
                margin: '0 0 0 10px'
            },
            // Add to overview ruler
            overviewRulerColor: 'blue',
            overviewRulerLane: vscode.OverviewRulerLane.Right
        });
        smellDecorationTypes.set('IM', decorationType);
    }
    return smellDecorationTypes.get('IM');
}



























function getStaticDecorationType(smellType) {
    if (!smellDecorationTypes.has(smellType)) {
        const decorationType = vscode.window.createTextEditorDecorationType({
            backgroundColor: 'rgba(255, 0, 0, 0.3)', // Red background for static
            border: '2px solid red',
            borderRadius: '3px',
            overviewRulerColor: 'red',
            overviewRulerLane: vscode.OverviewRulerLane.Right,
            after: {
                contentText: ` ⚠️ ${smellType}`,
                color: 'red',
                fontWeight: 'bold',
                margin: '0 0 0 10px'
            }
        });
        smellDecorationTypes.set(smellType, decorationType);
    }
    return smellDecorationTypes.get(smellType);
}

function clearHighlights(editor = null) {
    // Use provided editor or get active editor
    const targetEditor = editor || vscode.window.activeTextEditor;
    if (!targetEditor) return;
    
    // Clear all existing decorations
    smellDecorationTypes.forEach((decorationType) => {
        targetEditor.setDecorations(decorationType, []);
    });
}

// Function to dispose all decoration types (call this when deactivating extension)
function disposeDecorationTypes() {
    smellDecorationTypes.forEach((decorationType) => {
        decorationType.dispose();
    });
    smellDecorationTypes.clear();
}

// Enhanced version with more features
function highlightSmellsAdvanced(results, method) {
    clearHighlights();
    
    const editor = vscode.window.activeTextEditor;
    if (!editor) {
        vscode.window.showErrorMessage('No active editor found');
        return;
    }
    
    const smells = results.smells.smells.smells;
    const decorationsByType = new Map();
    
    // Show progress for large number of smells
    vscode.window.withProgress({
        location: vscode.ProgressLocation.Notification,
        title: "Highlighting Code Smells",
        cancellable: false
    }, async (progress) => {
        
        for (let i = 0; i < smells.length; i++) {
            const smell = smells[i];
            progress.report({ 
                increment: (100 / smells.length),
                message: `Processing smell ${i + 1} of ${smells.length}: ${smell.type}`
            });
            
            if (method === 'dynamic') {
                highlightDynamicSmell(smell, i, decorationsByType, editor);
            } else if (method === 'static') {
                highlightStaticSmell(smell, i, decorationsByType, editor);
            }
        }
        
        // Apply all decorations
        decorationsByType.forEach((ranges, decorationType) => {
            editor.setDecorations(decorationType, ranges);
        });
        
        // Show summary
        vscode.window.showInformationMessage(
            `Highlighted ${smells.length} code smell${smells.length !== 1 ? 's' : ''} using ${method} method`
        );
    });
}




































































// Function to call Python explain function
function callPythonExplain(filePath, smell, method) {

    console.log("Funzione callPythonExplain chiamata")

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



# Factory function to convert SimpleNamespace to appropriate class
def convert_to_smell_class(namespace_obj):
    smell_type = namespace_obj.type
    
    # Remove the 'type' attribute since it's not needed in the constructor
    namespace_dict = namespace_obj.__dict__.copy()
    smell_type = namespace_dict.pop('type').replace(" ","")

    #print("smell_type: ",smell_type)
    #print("type of smell_type: ",type(smell_type))
    
    # Map type strings to classes (you'll need to import your actual classes)
    class_map = {
        'IM': IM,
        'IQ': IQ,
        'IdQ': IdQ,
        'CG': CG,
        'LPQ': LPQ,
        'NC': NC,
        'ROC': ROC,
        'LC': LC,
        # Add other smell types here
        # 'OTHER_TYPE': OtherSmellClass,
    }
    
    if smell_type in class_map:
        return class_map[smell_type](**namespace_dict)
    else:
        return None



def serialize_explanation_result(obj):
    """Convert explanation result to JSON-serializable format"""
    if hasattr(obj, '__dict__'):
        result = {}
        for key, value in obj.__dict__.items():
            if not key.startswith('_'):
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
    # Add project root to Python path
    project_root = r"${projectRoot.replace(/\\/g, '\\\\')}"
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    #print("Importing Explainer and smell classes...")
    from smells.Explainer import Explainer

    from smells.IM.IMExplainer import IMExplainer
    from smells.IM.IM import IM  

    from smells.ROC.ROCExplainer import ROCExplainer
    from smells.ROC.ROC import ROC

    from smells.IQ.IQExplainer import IQExplainer
    from smells.IQ.IQ import IQ

    from smells.IdQ.IdQExplainer import IdQExplainer
    from smells.IdQ.IdQ import IdQ

    from smells.CG.CGExplainer import CGExplainer
    from smells.CG.CG import CG

    from smells.LC.LCExplainer import LCExplainer
    from smells.LC.LC import LC

    from smells.LPQ.LPQExplainer import LPQExplainer
    from smells.LPQ.LPQ import LPQ

    from smells.NC.NCExplainer import NCExplainer
    from smells.NC.NC import NC

    #print("Imports successful")

    # Initialize complete_explanation at the start
    complete_explanation = ""

    # Prepare the smell object
    smell_data = """${JSON.stringify(smell).replace(/"/g, '\\"')}"""
    smell_dict = json.loads(smell_data)
    #print(f"Smell dictionary: {smell_dict}")
    #print(f"Smell type: '{smell_dict.get('type', 'UNKNOWN')}'")
    #print(f"Method: '${method}'")

    with open(r'${filePath}', "r", encoding='utf-8') as f:
        code = f.read()
    

    # Debug the registry system
    #print("\\n=== Debugging Registry ===")
    #print(f"Explainer._explainers: {Explainer._explainers}")

    
    from types import SimpleNamespace
    smell_obj = SimpleNamespace(**smell_dict)

    #print("Smell Object")
    #print(smell_obj)
    
    # Convert to the appropriate smell class
    converted_obj = convert_to_smell_class(smell_obj)
    #print("\\nConverted Object:")
    #print(converted_obj)
    #print("\\nAs dict:")
    #print(converted_obj.as_dict())

    
    # Test get_explainer with the proper smell object
    #print("\\n=== Testing get_explainer ===")
    test_explainer = Explainer.get_explainer(code, converted_obj, '${method}')
    #print(f"get_explainer result: {test_explainer}")
    #print(f"Type: {type(test_explainer)}")
        
    #print("\\n=== Final Explanation Attempt ===")
    #print("Calling Explainer.explain with proper smell instance...")
    
    explanation_generator = Explainer.explain(code, converted_obj, '${method}')
    #print(f"Explainer.explain returned: {explanation_generator}")
    #print(f"Type: {type(explanation_generator)}")
    
    if explanation_generator:
        #print("Processing explanation generator...")
        chunk_count = 0
        for chunk in explanation_generator:
            if chunk:
                complete_explanation += chunk
                chunk_count += 1
                # Show progress every 20 chunks
                #if chunk_count % 20 == 0:
                    #print(f"  Processed {chunk_count} chunks, length so far: {len(complete_explanation)}")
        
        #print(f"Finished! Total chunks: {chunk_count}, Final length: {len(complete_explanation)}")
    else:
        complete_explanation = "No explanation generated - explainer not found or registered"
        #print("Explainer.explain returned None or empty generator")
    
    # Output the result (complete_explanation is now always defined)
    if complete_explanation and complete_explanation.strip():
        #print("\\nSUCCESS - outputting explanation")
        print(json.dumps({"explanation": complete_explanation, "success": True}))
    else:
        print(json.dumps({"explanation": "No explanation generated - check debug output above", "success": True}))
        #print("\\nNO EXPLANATION - outputting default message")
        
        
except Exception as e:
    #print(f"Exception occurred: {e}")
    import traceback
    error_details = traceback.format_exc()
    #print(f"Traceback: {error_details}")
    error_result = {
        "error": str(e), 
        "details": error_details,
        "success": False
    }
    print(json.dumps(error_result))`;

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