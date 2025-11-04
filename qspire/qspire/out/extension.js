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
/*
export function activate(context: vscode.ExtensionContext) {
  let disposable = vscode.commands.registerCommand('mytool.analyzeFile', async () => {
    // Step 1: Ask the user to pick a file from the workspace
    const uri = await vscode.window.showOpenDialog({
      canSelectMany: false,
      openLabel: 'Analyze This File',
      filters: {
        'Code files': ['js', 'py', 'ts', 'cpp', 'java', 'txt'] // adjust to your needs
      }
    });

    if (!uri || uri.length === 0) {
      vscode.window.showWarningMessage('No file selected.');
      return;
    }

    // Step 2: Read file content
    const filePath = uri[0].fsPath;
    const content = fs.readFileSync(filePath, 'utf8');

    // Step 3: Do something with it — for now, just show first 100 chars
    vscode.window.showInformationMessage(`First 100 characters:\n${content.slice(0, 100)}`);
  });

  context.subscriptions.push(disposable);
}
*/
let panel;
function activate(context) {
    const statusBarItem = vscode.window.createStatusBarItem(vscode.StatusBarAlignment.Left);
    statusBarItem.text = '$(graph) Analyzer';
    statusBarItem.command = 'mytool.toggleGui';
    statusBarItem.tooltip = 'Toggle Code Analyzer GUI';
    statusBarItem.show();
    context.subscriptions.push(statusBarItem);
    const toggleCommand = vscode.commands.registerCommand('mytool.toggleGui', () => {
        if (panel) {
            panel.dispose();
            panel = undefined;
        }
        else {
            panel = vscode.window.createWebviewPanel('codeAnalyzer', 'Code Analyzer GUI', vscode.ViewColumn.One, { enableScripts: true });
            const htmlPath = path.join(context.extensionPath, 'media', 'gui.html');
            const html = fs.readFileSync(htmlPath, 'utf8');
            panel.webview.html = html;
            panel.onDidDispose(() => {
                panel = undefined;
            });
            panel.webview.onDidReceiveMessage(async (message) => {
                if (message.command === 'analyze') {
                    // 1. Try to get the active editor
                    const editor = vscode.window.activeTextEditor;
                    if (editor && editor.document) {
                        const document = editor.document;
                        const filePath = document.uri.fsPath;
                        const content = document.getText();
                        vscode.window.showInformationMessage(`Using active file: ${filePath}`);
                        // Do your analysis here
                        console.log("Analyzing active file:", content.slice(0, 100));
                        // You can also send content back to the WebView if needed
                    }
                    else {
                        // 2. Ask the user to pick a file
                        const uri = await vscode.window.showOpenDialog({
                            canSelectMany: false,
                            openLabel: 'Choose file to analyze',
                            filters: {
                                'Code files': ['py'],
                            }
                        });
                        if (!uri || uri.length === 0) {
                            vscode.window.showWarningMessage('No file selected.');
                            return;
                        }
                        const filePath = uri[0].fsPath;
                        const content = fs.readFileSync(filePath, 'utf8');
                        vscode.window.showInformationMessage(`Using selected file: ${filePath}`);
                        console.log("Analyzing selected file:", content.slice(0, 100));
                    }
                }
            });
        }
    });
    context.subscriptions.push(toggleCommand);
}
/*
export function activate(context: vscode.ExtensionContext) {
  let disposable = vscode.commands.registerCommand('mytool.showGui', () => {
    const panel = vscode.window.createWebviewPanel(
      'codeAnalyzer',
      'Code Analyzer GUI',
      vscode.ViewColumn.One,
      {
        enableScripts: true
      }
    );

    const htmlPath = path.join(context.extensionPath, 'media', 'gui.html');
    const html = fs.readFileSync(htmlPath, 'utf8');
    panel.webview.html = html;
  });

  context.subscriptions.push(disposable);
}
*/ 
//# sourceMappingURL=extension.js.map