import { readFileSync } from 'node:fs';
import assert from 'node:assert/strict';

const src = readFileSync(new URL('../static/dashboard.jsx', import.meta.url), 'utf8');
const start = src.indexOf('function _provisionFreeBytes(');
const end = src.indexOf('\nfunction _serverBinaryVerdict', start);
const freeBytes = eval(`(${src.slice(start, end).trim()})`);

// Actual full-filesystem diagnostic from a new Dot in TWRP, including
// BusyBox's wrapped device name and reserved blocks (used < total).
assert.equal(freeBytes(`Filesystem           1K-blocks      Used Available Use% Mounted on
/dev/block/mmcblk0p16
                       1035032   1018648         0 100% /data`), 0);
assert.equal(freeBytes(`Filesystem 1K-blocks Used Available Use% Mounted on
/dev/block/mmcblk0p16 1035032 900000 118648 88% /data`), 118648 * 1024);
assert.equal(freeBytes(`Filesystem 1024-blocks Used Available Capacity Mounted on
/dev/block/mmcblk0p16 1035032 900000 118648 88% /data`), 118648 * 1024);
assert.equal(freeBytes('/sbin/sh: df: not found'), null);
assert.equal(freeBytes('Filesystem Size Used Free\n/data 1G 1G 0'), null);
assert.equal(freeBytes('Filesystem 1K-blocks Used Available Use% Mounted on'), null);
console.log('provision_storage: all checks passed.');
