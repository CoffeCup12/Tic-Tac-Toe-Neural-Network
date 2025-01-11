let board = document.querySelectorAll(".cell")
let button = document.getElementById("resetButton")
let players = ["X", "O"]
let currentPlayer = 0

async function move(index){

    board[index].innerText = players[currentPlayer]
    currentPlayer += 1

    const response = await fetch('/clicked',{
        method: "POST",
        headers: {
            "Content-Type" : "application/json"
        },
        body: JSON.stringify({content: index})
    })

    const result = await response.json()
    if("number" === typeof result){
        board[result].innerText = players[currentPlayer]
        currentPlayer -= 1
    }else{
        button.innerText = result
    }

}

async function reset(){
    
    const response = await fetch('/reset', {method: "GET"})
    for(let i = 0; i < 9; i++){
        board[i].innerText = ""
    }
    button.innerText = "Restart"
    currentPlayer = 0
}