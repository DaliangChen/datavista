import * as fs from 'fs';

const file = process.argv[2];

let content = fs.readFileSync(file, 'utf8');

content = content
    // .replace(/\\\( /g, '$')
    // .replace(/ \\\)/g, '$')
    // .replace(/\\\[ /g, '$$$')
    // .replace(/ \\\]/g, '$$$')
    // .replace(/\\\(/g, '$')
    // .replace(/\\\)/g, '$')
    // .replace(/\\\[/g, '$$$')
    // .replace(/\\\]/g, '$$$')
    // .replace(/\$([\u4e00-\u9fa5])/g, "$ $1")
    // .replace(/([\u4e00-\u9fa5])\$/g, "$1 \$")
    // .replace(/^#+\s*(.*)$/gm, '**$1**')
    // .replaceAll('\mathcal', '\mathscr')

fs.writeFileSync(file, content, 'utf8');