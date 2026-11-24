.PHONY: build package test clean

build:
	mkdir -p dist
	tinygo build -o dist/plugin.wasm -target=wasip1 -scheduler=none -buildmode=c-shared .

package: build
	python3 -c 'import zipfile; z=zipfile.ZipFile("dist/echora.ndp", "w", zipfile.ZIP_DEFLATED); z.write("manifest.json", "manifest.json"); z.write("dist/plugin.wasm", "plugin.wasm"); z.close()'

test:
	go test ./...
	cd contract-tests && go test ./...

clean:
	rm -f dist/plugin.wasm dist/echora.ndp
