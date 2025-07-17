import * as fs from 'fs';

const file = process.argv[2];

console.log(`Processing file: ${file}`);

if (file.toLocaleLowerCase().endsWith('preprocess.md')) {

    let content = fs.readFileSync(file, 'utf8');

    content = content
        .replace(/\\\( /g, '$')
        .replace(/ \\\)/g, '$')
        .replace(/\\\[ /g, '$$$')
        .replace(/ \\\]/g, '$$$')
        .replace(/\\\(/g, '$')
        .replace(/\\\)/g, '$')
        .replace(/\\\[/g, '$$$')
        .replace(/\\\]/g, '$$$')
        .replace(/\$([\u4e00-\u9fa5])/g, "$ $1")
        .replace(/([\u4e00-\u9fa5])\$/g, "$1 \$")
        .replace(/^#+\s*(.*)$/gm, '**$1**')
        .replaceAll('\mathcal', '\mathscr')

    fs.writeFileSync(file, content, 'utf8');
}

if (file.toLocaleLowerCase().endsWith('note\\math.md')) {

    let content = fs.readFileSync(file, 'utf8');

    content = content
        .replace(/。/g, '. ')
        .replace(/，/g, ', ')
        .replace(/）/g, ')')
        .replace(/（/g, '(')

    fs.writeFileSync(file, content, 'utf8');
}
